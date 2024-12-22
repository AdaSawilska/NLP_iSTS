import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    inputs = [
        f"[CLS] {row['x1']} [SEP] {row['x2']} [SEP] {row['sentence1']} [SEP] {row['sentence2']} [SEP]"
        for _, row in examples.iterrows()
    ]
    tokenized_inputs = tokenizer(inputs,
                                 padding=True,
                                 truncation=True,
                                 max_length=256)
    tokenized_inputs['labels'] = examples['y_score'].tolist()  # Add regression labels
    return tokenized_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()  # Adjust for regression output
    return {"mse": mean_squared_error(labels, predictions)}


train_path = 'data/Semeval2016/train/train_healines.csv'
valid_path = 'data/Semeval2016/train/validation_healines.csv'
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

train_dataset = train_dataset.map(lambda x: preprocess_function(train_df), batched=False)
valid_dataset = valid_dataset.map(lambda x: preprocess_function(valid_df), batched=False)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)
# model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    remove_unused_columns=False,
    gradient_accumulation_steps=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
results = trainer.evaluate(eval_dataset=valid_dataset)
print(results)

# Save model and tokenizer
model.save_pretrained("./trained_roberta")
tokenizer.save_pretrained("./trained_roberta")
