فيما يلي «خرائط طريق» شاملة لبناء نموذج لغويٍّ ذكي (LLM) من الصفر يُطلق عليه **Cosmos**، ثم تدريبه بطريقة «ذكيّة» (Efficient/Adaptive) بحيث يمكن استدعاؤه لاحقًا عبر:

```python
transformers.pipeline("text-generation", model="cosmos-lm", ...)
```

الملف التالي يتضمّن:

1. أهداف التصميم (المعايير)
2. هندسة البيانات (Data-Centric)
3. هندسة النموذج (Model-Centric)
4. منهج التدريب الذكي (Smart-Training)
5. نصوص كاملة (PyTorch + HuggingFace) لكل مرحلة
6. إجراءات التقييم والضبط الدقيق
7. نصائح الإنتاج (Deployment)

---

### 1) معايير التصميم (Design Specs)

| البند | الهدف |
|-------|--------|
| الحجم | 1–3B متغيّر (قابل للتوسّع) |
| السياق | 4K → 8K توكن |
| اللغات | إنجليزي + عربي (يمكن التوسّع) |
| دقّة التحويل | bfloat16 (Ampere+) |
| التوازي | DeepSpeed ZeRO-3 + FSDP |
| التدريب الذكي | Curriculum, Dynamic Sampling, Early-Exit, LoRA/AdaLoRA |
| الأخلاقيات & الأمان | مطابقة OpenAI Moderation API + detoxify |
| الإصدار | نسخة «محادثة» (chat) تُحفِّز system/user/assistant |

---

### 2) هندسة البيانات (Data-Centric)

#### 2.1 مصادر أساسية (مجانية)
- **English**: SlimPajama, FineWeb-Edu, StarCoder, C4
- **Arabic**: Arabic-OSCAR, AraBERT-News, WikiArabia
- **Code**: Stack-v2 (filter 1→5 stars)
- **Instruct**: OpenHermes, Aya, SlimOrca-Arabic

#### 2.2 تنظيف عام (Python)
```python
# data_clean.py
import json, re, ftfy, emoji, gzip
from detoxify import Detoxify
detox = Detoxify('original', device='cuda')

url_re = re.compile(r'http\S+')
ref_re = re.compile(r'\[\d+\]|\(\d{4}\)')
emoji_re = emoji.get_emoji_regexp()

def clean(text):
    text = ftfy.fix_text(text)
    text = url_re.sub('', text)
    text = ref_re.sub('', text)
    text = emoji_re.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe(text, thresh=0.9):
    scores = detox.predict(text)
    return all(v < thresh for v in scores.values())

def process(src, dst):
    with gzip.open(src, 'rt') as fi, gzip.open(dst, 'wt') as fo:
        for line in fi:
            d = json.loads(line)
            d['text'] = clean(d['text'])
            if d['text'] and safe(d['text']):
                fo.write(json.dumps(d, ensure_ascii=False)+'\n')
```

#### 2.3 توليد «محادثات» صناعية (Self-Instruct)
نستخدم نموذجًا قائمًا (مثل Llama-3-8B) لتوليف أسئلة/أجوبة عالية الجودة ثم نراجعها بالنموذج نفسه (Reflexion):

```python
# synth_chat.py
from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct",
                torch_dtype=torch.bfloat16, device_map="auto")

SYS = "You are Cosmos, an Arabic/English AI assistant. Generate a helpful, concise answer."
def make_conv(prompt):
    messages = [{"role": "system", "content": SYS},
                {"role": "user",   "content": prompt}]
    out = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7)
    return out[0]['generated_text'][-1]['content']

topics = load_arabic_topics()  # ~20k
with open("synth_conv.jsonl", "w") as f:
    for t in topics:
        q = f"اكتب سؤالاً عن {t} ثم أجب عليه."
        conv = make_conv(q)
        f.write(json.dumps({"text": conv}, ensure_ascii=False)+"\n")
```

#### 2.4 تجميع المزيج (Mix-of-Domains)
نحافظ على نسب (Tokens):
- 60% متن عام
- 15% شفرة
- 15% تعليمات (instruct)
- 10% لغات غير إنجليزية (غالبًا عربي)

نحوّل كل شيء إلى تنسيق ChatML:

```
<|im_start|>system
You are Cosmos senior genius developer agent!<|im_end|>
<|im_start|>user
Who are you?<|im_end|>
<|im_start|>assistant
I am Cosmos, an AI assistant … <|im_end|>
```

---

### 3) هندسة النموذج (Model-Centric)

#### 3.1 هيكل GPT-NeoX (Decoder-Only)
- Layers = 24 (1.3B) أو 30 (2.7B)
- Hidden = 2560 / 3200
- Heads = 32 / 32, Head-Dim = 80
- RoPE θ=10K, Context = 4096 (قابل للتوسيع إلى 8K عبر PI)
- SwiGLU MLP
- RMSPreNorm
- Embedding tying
- Vocab = 64K (BPE merges على مزيج إنجليزي/عربي/شفرة)

#### 3.2 كود تعريف النموذج (PyTorch)
```python
# model.py
import math, torch, torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        dtype = x.dtype
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype) * self.weight

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_pos=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_pos = max_pos
        self._set_cos_sin_cache(max_pos)
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    def forward(self, x, seq_len):
        if seq_len > self.max_pos:
            self._set_cos_sin_cache(seq_len)
        cos, sin = self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        return (x * cos) + (self.rotate_half(x) * sin)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, 3*config.hidden_size, bias=False)
        self.o   = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_pos=config.max_position_embeddings)
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).transpose(1,3)
        q,k,v = qkv.unbind(2)
        q, k = map(lambda t: self.rotary(t, T), (q, k))
        att = (q @ k.transpose(-2,-1)) * self.scale
        if mask is not None:
            att = att.masked_fill(mask==0, float('-inf'))
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.o(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = int(8/3 * config.hidden_size)  # SwiGLU
        self.gate = nn.Linear(config.hidden_size, h, bias=False)
        self.up   = nn.Linear(config.hidden_size, h, bias=False)
        self.down = nn.Linear(h, config.hidden_size, bias=False)
    def forward(self, x):
        return self.down( nn.functional.silu(self.gate(x)) * self.up(x) )

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp  = MLP(config)
        self.ln1  = RMSNorm(config.hidden_size)
        self.ln2  = RMSNorm(config.hidden_size)
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class CosmosConfig(PretrainedConfig):
    model_type = "cosmos"
    def __init__(self, vocab_size=64000, hidden_size=2560, n_layers=24,
                 n_heads=32, max_position_embeddings=4096, **kw):
        super().__init__(**kw)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings

class CosmosModel(PreTrainedModel):
    config_class = CosmosConfig
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tying
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def get_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device))==1
    def forward(self, input_ids, labels=None, **kw):
        B, T = input_ids.size()
        x = self.embed(input_ids)
        mask = self.get_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                               shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)
```

---

### 4) تدريب ذكي (Smart-Training)

#### 4.1 منهج تعليمي تدريجي (Curriculum)
نبدأ بسياق 1K → 2K → 4K توكن تدريجيًا (Position Interpolation).

#### 4.2 عيّنة ديناميكية (Dynamic Sampling)
نحسب loss-per-domain كل خطوة، ونعيّن weight ∝ exp(−loss/T).

#### 4.3 Early-Exit (اختياري)
نضع رأس تصنيف إضافيًا بعد Layer-12؛ إذا انخفض entropy < threshold نخرج مبكرًا ونكافئ بالـ loss المُقلَّص.

#### 4.4 DeepSpeed ZeRO-3 config
```json
{
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_param": {"device": "cpu"},
    "offload_optimizer": {"device": "cpu"}
  },
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "steps_per_print": 10
}
```

#### 4.5 كود التدريب
```python
# train.py
import os, torch, transformers, deepspeed
from datasets import load_dataset
from model import CosmosModel, CosmosConfig
from tokenizer import load_tokenizer  # SentencePiece 64K

tok = load_tokenizer("cosmos-spm-64k")
config = CosmosConfig(vocab_size=tok.vocab_size(),
                      hidden_size=2560,
                      n_layers=24,
                      max_position_embeddings=4096)

model = CosmosModel(config)
model.gradient_checkpointing_enable()

data = load_dataset("json", data_files={"train": "train-chatml.jsonl"})
def encode(ex):
    ids = tok(ex["text"], truncation=True, max_length=2048)
    ids["labels"] = ids["input_ids"].copy()
    return ids
data = data.map(encode, batched=True, remove_columns=data["train"].column_names)

training_args = transformers.TrainingArguments(
    output_dir="cosmos-lm-1b3",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    warmup_steps=2000,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    deepspeed="ds_config_zero3.json",
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    report_to="tensorboard"
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    tokenizer=tok
)
trainer.train()
trainer.save_model("cosmos-lm-1b3/final")
tok.save_pretrained("cosmos-lm-1b3/final")
```

#### 4.6 استكمال التحبير (Instruction-Tuning) عبر LoRA
```python
# lora_tuning.py
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(r=64, lora_alpha=128, target_modules=["qkv","o","gate","up","down"])
lora_model = get_peft_model(model, lora_cfg)
# ――> نعيد التدريب على بيانات التعليمات فقط (lr=1e-4, 2000 خطوة)
```

---

### 5) تقييم سريع (Evaluation)

- **Perplexity**: WikiText-2 / AraTest
- **Down-Stream**: ARC-E, HellaSwag (إنجليزي) + ALUE (عربي)
- **Code**: HumanEval (pass@1)
- **الأمان**: ToxiGen, AAI (Arabic-Adversarial-Inputs)

نستخدم lm-eval-harness:

```bash
lm_eval --model hf --model_args pretrained=cosmos-lm-1b3 --tasks arc_easy,hellaswag,arabic_mmlu --batch_size 16
```

---

### 6) دمج النموذج في pipeline
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "cosmos-lm-1b3"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tok,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7)

messages = [
    {"role": "system", "content": "You are a Cosmos senior genius developer agent!"},
    {"role": "user",   "content": "Who are you?"}
]
print(pipe(messages)[0]['generated_text'][-1]['content'])
```

---

### 7) نصائح الإنتاج (Deployment)

- استخدم Text-Generation-Inference (TGI) أو vLLM لخدمة سياقات أطول بكفاءة.
- قم بتحويل الوزن إلى Safetensors.
- استخدم quantization (GPTQ 4bit / AWQ) لتقليل الحجم 75%.
- وضع cache بـ Redis لإعادة الاستفسارات المتشابهة.
- وضع فلترة سريعة (detoxify) قبل إرجاع الرد للمستخدم.
- سجّل سجلات المحادثات لتحليل الانحراف (drift) وإعادة التدريب الدوري.

---

### خلاصة
1. نظّف وبنِ بياناتك بعناية (Data-Centric).
2. صمّم نموذجك من الصفر أو عدّل نموذجًا مفتوحًا (Model-Centric).
3. طبّق تقنيات تدريب ذكية (Curriculum, Dynamic Sampling, LoRA, Early-Exit).
4. قيّم واطبق معايير الأمان.
5. حوّل إلى HuggingFace format ليعمل مع pipeline مباشرة.

بهذه الخطوات تكون قد أنشأت نموذج Cosmos الخاص بك، ودربته بذكاء، وجهّزته للاستدعاء بالطريقة التي وردت في سؤالك. حظًا موفقًا!
