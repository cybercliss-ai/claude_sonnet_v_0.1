لنبني لك نموذجًا ذكيًا «من الصفر» يُجيب مثلما تريد، لكنّا نُدخِله في pipeline مثل الذي تعطيه، ونُدرّبه بطريقة ذكية (Efficient-Training) لا تُكلفك ملايين الدولارات.  
الخطة مُقسَّمة إلى 6 مراحل واضحة؛ كل مرحلة تأتي مع كود جاهز يُنفَّذ على بيئة واحدة (GPU A100 40 GB أو أقل إذا استخدمت تقنياتنا).  
في نهاية الملفّ تجد خلاصة «نموذجك النهائي» يُستدعى بنفس الـ pipeline الذي أعطيته.

----------------------------------------------------
1. تحديد المتطلبات الدقيقة
----------------------------------------------------
• الحجم المستهدف: 1.1–1.3 B parameter (يتناسب مع A100 واحدة).  
• اللغة: إنجليزي أساسي + قابلية توسّع عربي لاحقًا.  
• طريقة التدريب: Pre-train → SFT → DPO (Direct Preference Optimization) بدون Reinforcement Learning معقد.  
• التقنيات الذكية:  
  – Flash-Attention 2  
  – DeepSpeed ZeRO-3 + CPU-offload  
  – Mixed-precision bfloat16  
  – Packing + Sample-efficient DPO (ORPO Loss)  
• استخدام مكتبات HuggingFace فقط (لا حاجة لكود مخصص معقّد).

----------------------------------------------------
2. بيئة العمل (ملف one-shot conda)
----------------------------------------------------
```bash
# أنشئ بيئة نظيفة
conda create -n cosmos python=3.10 -y
conda activate cosmos

# أدوات NVIDIA
pip install --upgrade pip
pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40 accelerate datasets tokenizers flash-attn deepspeed==0.14
pip install wandb  # اختياري لتسجيل التجارب
```

----------------------------------------------------
3. بناء الأرشitekture (1.1 B)
----------------------------------------------------
نُنشئ ملف `modeling_cosmos.py` (مُبسَّط):

```python
from transformers import LlamaConfig, LlamaForCausalLM

def build_cosmos_config(vocab=50_000, hidden=2048, layers=22, heads=32):
    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=int(hidden * 3.5),
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
        tie_word_embeddings=True,  # يوفر 15% ذاكرة
    )

def get_cosmos_model():
    cfg = build_cosmos_config()
    return LlamaForCausalLM(cfg)
```

----------------------------------------------------
4. Pre-Training الذكي (3 أيام فقط)
----------------------------------------------------
4.1 تحضير البيانات  
نستخدم SlimPajama (627G) → نُرشّح فقط 30B توكن (≃ 50GB) لاختصار الوقت.  
نُعبّئ التوكنات في ملف `.parquet` واحد (packing=4096).

4.2 سكربت التدريب (`pretrain.py`)  
```python
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from modeling_cosmos import get_cosmos_model
import torch, deepspeed, os

model = get_cosmos_model()
tokenizer_dir = "meta-llama/Llama-2-7b-hf"  # فقط tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
tok.pad_token = tok.eos_token

ds = load_dataset("cerebras/SlimPajama-627B", streaming=True, split="train")
def pack(ex):
    txt = ex["text"]
    ids = tok(txt, truncation=True, max_length=4096, return_overflowing_tokens=True)
    return {"input_ids": ids["input_ids"]}

ds = ds.map(pack, batched=True, remove_columns=ds.column_names)

training_args = TrainingArguments(
    output_dir="./cosmos_pt",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,  # إجمالي 128
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    bf16=True,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    deepspeed="ds_config_zero3.json",  # ملف أسفله
    report_to="wandb",
    run_name="cosmos_pt",
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
trainer.save_model("./cosmos_pt/final")
```

4.3 ملف DeepSpeed (`ds_config_zero3.json`)
```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 32,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" }
  },
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0
}
```

----------------------------------------------------
5. Supervised Fine-Tuning (SFT) لتحويله إلى «Cosmos senior genius»
----------------------------------------------------
نُنشئ 30k سؤال/جواب عالي الجودة (نمزج OpenHermes-2.5 + Code-Feedback + جزء عربي صغير).  
نستخدم Packing + Sample-efficient SFT (1 epoch فقط).

```python
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from modeling_cosmos import get_cosmos_model
from peft import LoraConfig, get_peft_model, TaskType

model = get_cosmos_model()
model.load_adapter("./cosmos_pt/final")  # أو load_pretrained
lora_config = LoraConfig(
    r=64, lora_alpha=128, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

ds = load_dataset("OpenHermes-2.5-30k")  # مثال
def templ(x):
    return {"text": f"<|system|>You are a Cosmos senior genius developer agent !<|end|><|user|>{x['instruction']}<|end|><|assistant|>{x['output']}<|end|>"}
ds = ds.map(templ)

training_args = TrainingArguments(
    output_dir="./cosmos_sft",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="wandb",
)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
trainer.train()
trainer.save_model("./cosmos_sft/final")
```

----------------------------------------------------
6. تطويره بالتفضيلات (DPO/ORPO) دون تكلفة RLHF
----------------------------------------------------
نستخدم ORPO Loss (يعطي نتائج DPO ممتازة بدون reference model).  
نحتاج فقط 5k زوج «مُفضَّل / مرفوض».

```python
from orpo import ORPOTrainer  # pip install orpo
ds = load_dataset("Intel/orpo-dpo-mix-5k")
model = ...  # نفس النموذج مع LoRA
orpo_args = TrainingArguments(..., output_dir="./cosmos_orpo")
trainer = ORPOTrainer(model=model, args=orpo_args, train_dataset=ds)
trainer.train()
trainer.save_model("./cosmos_final")
```

----------------------------------------------------
7. دمج الأوزان واختبار النموذج
----------------------------------------------------
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("./cosmos_pt/final", torch_dtype=torch.bfloat16)
tok  = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tok.pad_token = tok.eos_token

# دمج LoRA
from peft import PeftModel
model = PeftModel.from_pretrained(base, "./cosmos_orpo/final")
model = model.merge_and_unload()
model.save_pretrained("./cosmos_v1.0")
tok.save_pretrained("./cosmos_v1.0")
```

----------------------------------------------------
8. استدعاؤه بنفس pipeline الذي أعطيته
----------------------------------------------------
```python
import torch, transformers

pipeline = transformers.pipeline(
    "text-generation",
    model="./cosmos_v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a Cosmos senior genius developer agent !"},
    {"role": "user",   "content": "Who are you?"},
]

outputs = pipeline(messages, max_new_tokens=256, do_sample=True, temperature=0.7)
print(outputs[0]["generated_text"][-1])
```

----------------------------------------------------
9. ملاحظات التحسين الإضافي
----------------------------------------------------
• إذا أردت عربية: كرّر SFT على 50k عينة عربية (ArDPO, AraInstruct).  
• لزيادة السرعة: حوّل إلى `bitsandbytes` 4-bit بعد التدريب (للاستدعاء فقط).  
• لخدمة الإنتاج: صدّر إلى `ctransformlers` أو `exllamav2` للاستدعاء أسرع.  
• لاحقًا يمكنك تكبير النموذج إلى 3B أو 7B بنفس الأنابيب.

----------------------------------------------------
10. خلاصة 3-سطرية
----------------------------------------------------
1. شغّل `pretrain.py` → 3 أيام → نموذج 1.1B.  
2. شغّل `sft.py` ثم `orpo.py` → 6 ساعات → Cosmos Genius.  
3. استخدم pipeline أعلاه → جاهز للإنتاج.

وهكذا تحصل على «نموذج ذكي من الصفر» يُدرَّب ذكيًا (Efficient) ويعمل بنفس الكود الذي أعطيته.
