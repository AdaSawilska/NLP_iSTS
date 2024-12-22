import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import DataCollatorWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    # Create text pairs by combining chunks and sentences
    texts = [
        f"{str(x1)} </s></s> {str(x2)} </s></s> {str(s1)} </s></s> {str(s2)}"
        for x1, x2, s1, s2 in zip(
            examples['x1'],
            examples['x2'],
            examples['sentence1'],
            examples['sentence2']
        )
    ]

    # Tokenize all texts at once
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None
    )

    # Convert scores to integer labels
    tokenized["labels"] = [int(score) for score in examples["y_score"]]

    return tokenized


def prepare_dataset(file_path):
    # Read CSV
    df = pd.read_csv(file_path, sep=';')

    # Clean and prepare data
    df["x1"] = df["x1"].fillna("EMPTY").astype(str)
    df["x2"] = df["x2"].fillna("EMPTY").astype(str)
    df["sentence1"] = df["sentence1"].fillna("EMPTY").astype(str)
    df["sentence2"] = df["sentence2"].fillna("EMPTY").astype(str)
    df["y_score"] = df["y_score"].replace("NIL", 0).astype(float)
    df['y_score'] = pd.to_numeric(df['y_score'], errors='coerce')
    df.dropna(subset=['y_score'], inplace=True)

    # Round float scores to nearest integer if necessary
    df['y_score'] = df['y_score'].round().astype(int)

    # Convert to dataset
    dataset = Dataset.from_pandas(df)

    # Apply preprocessing
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return processed_dataset


# Prepare datasets
train_dataset = prepare_dataset('data/Semeval2016/train/training.csv')
valid_dataset = prepare_dataset('data/Semeval2016/train/validation.csv')

# Initialize model for classification
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=6,  # 0 to 5 inclusive
    problem_type="single_label_classification"
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_weighted": f1_score(labels, predictions, average='weighted')
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print(results)

# Save model and tokenizer
model.save_pretrained("./trained_roberta")
tokenizer.save_pretrained("./trained_roberta")