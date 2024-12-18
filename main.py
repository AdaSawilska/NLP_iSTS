from transformers import AutoTokenizer
import pandas as pd
from transformers import RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import mean_squared_error


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    # Combine x1, x2, sentence1, sentence2 for context-aware training
    inputs = [
        f"{row['x1']} [SEP] {row['x2']} [CLS] {row['sentence1']} [SEP] {row['sentence2']}"
        for _, row in examples.iterrows()
    ]
    labels = examples['y_score'].tolist()  # For regression
    return tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"), labels



train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")
train_data = preprocess_function(train_df)
valid_data = preprocess_function(valid_df)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=1)  # Regression

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()  # Adjust for regression output
    return {"mse": mean_squared_error(labels, predictions)}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate(eval_dataset=valid_dataset)
print(results)

model.save_pretrained("./trained_roberta")
tokenizer.save_pretrained("./trained_roberta")