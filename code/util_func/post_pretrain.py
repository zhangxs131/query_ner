import torch
from transformers import AutoConfig,AutoTokenizer,AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os



model_path=''
data_path=''

# 配置RoBERTa模型
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer=AutoTokenizer.from_pretrained(model_path)


# 配置数据集
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=data_path,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 进行post-pretrain训练
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    save_steps=100000,
    save_total_limit=2,
    prediction_loss_only=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)
trainer.train()
