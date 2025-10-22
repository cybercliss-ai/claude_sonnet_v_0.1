# claude_sonnet_v_0.1
# 🚀 **الدليل الشامل من التحميل إلى التدريب إلى التعديل**

## 📥 **الجزء 1: التحميل والإعداد**

### **1.1 إعداد البيئة**
```bash
# تثبيت المكتبات الأساسية
pip install transformers torch accelerate datasets peft bitsandbytes
pip install trl wandb sentencepiece protobuf

# للتجارب المتقدمة
pip install flash-attn --no-build-isolation
```

### **1.2 تحميل النموذج الأساسي**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# تحميل النموذج مع تحسين الذاكرة
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,  # لتحسين الذاكرة
    trust_remote_code=True
)
```

## 🔧 **الجزء 2: التعديلات داخل النموذج**

### **2.1 إعداد النموذج للتدريب المتقدم**
```python
# تحضير النموذج للتدريب بالبتات المنخفضة
model = prepare_model_for_kbit_training(model)

# تكوين LoRA للتعديل الفعال
lora_config = LoraConfig(
    r=16,  # رتبة التعديل
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# تطبيق LoRA على النموذج
model = get_peft_model(model, lora_config)
```

### **2.2 إضافة طبقات مخصصة**
```python
import torch.nn as nn

class CustomAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return self.layer_norm(x + attended)

# إضافة طبقة مخصصة للنموذج
def add_custom_layers(model, num_layers=2):
    for i in range(num_layers):
        custom_layer = CustomAttentionLayer(model.config.hidden_size)
        # إدراج الطبقة المخصصة في النموذج
        model.model.layers.insert(-1, custom_layer)
    return model

# تطبيق التعديل
model = add_custom_layers(model)
```

## 🎯 **الجزء 3: طرق إضافة الخصائص العملية**

### **3.1 نظام الأدوار (Role System) مثل Claude**
```python
def create_claude_like_prompt(messages):
    """
    إنشاء نظام أدوار مشابه لـ Claude
    """
    system_prompt = """أنت مساعد ذكي ومفيد. يجب أن:
    - تكون دقيقاً في المعلومات
    - تكون متوازناً في الردود
    - تقدم تفسيرات منطقية
    - تعترف عندما لا تعرف الإجابة"""
    
    formatted = f"<|system|>\n{system_prompt}\n<|end|>\n"
    
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<|user|>\n{msg['content']}\n<|end|>\n"
        else:
            formatted += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
    
    return formatted + "<|assistant|>\n"

# اختبار النظام
test_messages = [
    {"role": "user", "content": "ما هو الذكاء الاصطناعي؟"}
]
prompt = create_claude_like_prompt(test_messages)
```

### **3.2 إضافة ذاكرة السياق**
```python
class ContextMemory:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.conversation_history = []
    
    def add_interaction(self, user_input, assistant_response):
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response,
            "tokens": len(user_input.split()) + len(assistant_response.split())
        })
        self._trim_history()
    
    def _trim_history(self):
        total_tokens = sum(item["tokens"] for item in self.conversation_history)
        while total_tokens > self.max_tokens and self.conversation_history:
            removed = self.conversation_history.pop(0)
            total_tokens -= removed["tokens"]
    
    def get_context(self):
        context = ""
        for interaction in self.conversation_history[-5:]:  # آخر 5 تفاعلات
            context += f"User: {interaction['user']}\nAssistant: {interaction['assistant']}\n"
        return context

# استخدام الذاكرة
memory = ContextMemory()
```

## 📊 **الجزء 4: إعداد البيانات للتدريب**

### **4.1 تحضير مجموعة البيانات**
```python
from datasets import Dataset, load_dataset
import json

def load_training_data(file_path):
    """تحميل بيانات التدريب من ملف JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # تنسيق كل عينة تدريب
        prompt = create_claude_like_prompt([
            {"role": "user", "content": item["instruction"]}
        ])
        
        formatted_data.append({
            "text": prompt + item["output"] + tokenizer.eos_token,
            "prompt": prompt
        })
    
    return Dataset.from_list(formatted_data)

# تحميل البيانات
dataset = load_training_data("training_data.json")
```

### **4.2 معالجة البيانات للنموذج**
```python
def tokenize_function(examples):
    """دالة لمعالجة النصوص للنموذج"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=2048,
        return_offsets_mapping=False
    )
    
    # إنشاء labels (نفس input_ids للتدريب على التوليد)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# تطبيق المعالجة
tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## 🏋️ **الجزء 5: التدريب المتقدم**

### **5.1 إعدادات التدريب المتقدمة**
```python
training_args = TrainingArguments(
    output_dir="./claude-like-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=3,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="wandb",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="paged_adamw_32bit"
)
```

### **5.2 تدريب النموذج مع callbacks مخصصة**
```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A')}")
    
    def on_save(self, args, state, control, **kwargs):
        print(f"تم حفظ النموذج في الخطوة {state.global_step}")

# إنشاء المدرب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    callbacks=[CustomCallback()]
)

# بدء التدريب
print("بدء التدريب...")
trainer.train()
```

## 🔄 **الجزء 6: طرق التعديل المتقدمة**

### **6.1 تعديل آلية الانتباه**
```python
def modify_attention_mechanism(model):
    """تعديل آلية الانتباه لتحسين الأداء"""
    for layer in model.model.layers:
        # تعديل معاملات الانتباه
        original_attention = layer.self_attn
        
        # إضافة انتباه متقاطع
        layer.cross_attention = nn.MultiheadAttention(
            embed_dim=model.config.hidden_size,
            num_heads=model.config.num_attention_heads
        )
    
    return model

# تطبيق التعديل
model = modify_attention_mechanism(model)
```

### **6.2 إضافة آلية التفكير المتسلسل**
```python
def add_chain_of_thought_capability(model, tokenizer):
    """إضافة قدرة التفكير المتسلسل"""
    
    def chain_of_thought_generate(prompt, max_length=512):
        # إضافة تعليمات للتفكير المتسلسل
        cot_prompt = prompt + "\nدعنا نفكر خطوة بخطوة:"
        
        inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # إضافة الدالة للنموذج
    model.chain_of_thought_generate = chain_of_thought_generate
    
    return model

# تطبيق آلية التفكير المتسلسل
model = add_chain_of_thought_capability(model, tokenizer)
```

## 💾 **الجزء 7: الحفظ والتقييم**

### **7.1 حفظ النموذج المدرب**
```python
# حفظ النموذج النهائي
trainer.save_model("./claude-like-model-final")

# حفظ إعدادات LoRA بشكل منفصل
model.save_pretrained("./claude-like-lora-adapters")

# حفظ التوكنايزر
tokenizer.save_pretrained("./claude-like-model-final")
```

### **7.2 تحميل النموذج المدرب للاستخدام**
```python
def load_trained_model(model_path):
    """تحميل النموذج المدرب للاستخدام"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# الاستخدام
model, tokenizer = load_trained_model("./claude-like-model-final")
```

## 🧪 **الجزء 8: الاختبار والتقييم**

### **8.1 دالة اختبار شاملة**
```python
def test_model_comprehensive(model, tokenizer, test_cases):
    """اختبار شامل للنموذج"""
    results = []
    
    for i, test_case in enumerate(test_cases):
        prompt = create_claude_like_prompt([
            {"role": "user", "content": test_case}
        ])
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "test_case": test_case,
            "response": response,
            "prompt": prompt
        })
        
        print(f"الاختبار {i+1}: {test_case}")
        print(f"الرد: {response}\n")
    
    return results

# حالات اختبار متنوعة
test_cases = [
    "اشرح نظرية النسبية بشكل مبسط",
    "كيف يمكن تحسين التعليم في الوطن العربي؟",
    "اكتب قصة قصيرة عن رحلة إلى الفضاء"
]

# تشغيل الاختبارات
results = test_model_comprehensive(model, tokenizer, test_cases)
```

## 📝 **نصائح نهائية مهمة:**

1. **ابدأ صغيراً**: جرب مع النماذج الصغيرة أولاً
2. **احتفظ بنسخ احتياطية**: احفظ النموذج بعد كل مرحلة تدريب مهمة
3. **راقب الأداء**: استخدم أدوات مثل wandb للمراقبة
4. **اختبر باستمرار**: اختبر النموذج بعد كل دورة تدريب
5. **حسن التوقفات**: استخدم Early Stopping لتجنب Overfitting

هذا الدليل يغطي المسار الكامل من البداية إلى النهاية. أي جزء تريد أن أركز عليه أكثر أو تشرحه بتفصيل إضافي؟
