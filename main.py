import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import DataCollatorWithPadding
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput

# Model
class RobertaForMultiTaskClassification(nn.Module):
    def __init__(self, model_name, num_score_labels=6, num_type_labels=7):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.score_projection = nn.Linear(hidden_size, hidden_size)
        self.type_projection = nn.Linear(hidden_size, hidden_size)

        self.score_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_score_labels)
        )
        self.type_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_type_labels)
        )
        self.log_vars = nn.Parameter(torch.zeros(2))

    def gradient_checkpointing_enable(self, **kwargs):
        self.roberta.gradient_checkpointing_enable()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            type_labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state[:, 0, :]

        score_features = self.score_projection(sequence_output)
        type_features = self.type_projection(sequence_output)

        score_logits = self.score_classifier(sequence_output)
        type_logits = self.type_classifier(sequence_output)

        loss = None
        if labels is not None and type_labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            score_loss = loss_fct(score_logits.view(-1, score_logits.size(-1)), labels.view(-1))
            type_loss = loss_fct(type_logits.view(-1, type_logits.size(-1)), type_labels.view(-1))

            # Apply learned weights with uncertainty
            precision1 = torch.exp(-self.log_vars[0])
            loss1 = precision1 * score_loss + self.log_vars[0]
            precision2 = torch.exp(-self.log_vars[1])
            loss2 = precision2 * type_loss + self.log_vars[1]
            loss = torch.mean(loss1 + loss2)

        return SequenceClassifierOutput(
            loss=loss,
            logits=(score_logits, type_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, path):
        self.roberta.save_pretrained(path)
        torch.save({
            'score_projection': self.score_projection.state_dict(),
            'type_projection': self.type_projection.state_dict(),
            'score_classifier': self.score_classifier.state_dict(),
            'type_classifier': self.type_classifier.state_dict(),
            'log_vars': self.log_vars
        }, f"{path}/classification_heads.pt")

    def load_pretrained(self, path):
        self.roberta = AutoModel.from_pretrained(path)
        checkpoint = torch.load(f"{path}/classification_heads.pt")
        self.score_projection.load_state_dict(checkpoint['score_projection'])
        self.type_projection.load_state_dict(checkpoint['type_projection'])
        self.score_classifier.load_state_dict(checkpoint['score_classifier'])
        self.type_classifier.load_state_dict(checkpoint['type_classifier'])
        self.log_vars = checkpoint['log_vars']


# Preprocess for RoBeRta
def preprocess_function(examples):
    texts = [
        f"{str(x1)} </s></s> {str(x2)} </s></s> {str(s1)} </s></s> {str(s2)}"
        for x1, x2, s1, s2 in zip(
            examples['x1'],
            examples['x2'],
            examples['sentence1'],
            examples['sentence2']
        )
    ]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None
    )
    tokenized["labels"] = [int(score) for score in examples["y_score"]]
    tokenized["type_labels"] = [int(type_) for type_ in examples["y_type"]]

    return tokenized


def prepare_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t')

    df["x1"] = df["x1"].fillna("EMPTY").astype(str)
    df["x2"] = df["x2"].fillna("EMPTY").astype(str)
    df["sentence1"] = df["sentence1"].fillna("EMPTY").astype(str)
    df["sentence2"] = df["sentence2"].fillna("EMPTY").astype(str)

    df["y_score"] = df["y_score"].replace("NIL", 0).astype(float)
    df['y_score'] = pd.to_numeric(df['y_score'], errors='coerce')
    type_mapping = {'EQUI': 0, 'OPPO': 1, 'SPE1': 2, 'SPE2': 3,  'SIMI':4, 'REL':5, 'NOALI':6}
    df['y_type'] = df['y_type'].map(type_mapping)
    df.dropna(subset=['y_score', 'y_type'], inplace=True)
    df['y_score'] = df['y_score'].round().astype(int)

    dataset = Dataset.from_pandas(df)
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    print_distribution_stats(df)


    return processed_dataset
def print_distribution_stats(df):
    """Print distribution statistics for the dataset"""
    print(f"Total samples: {len(df)}")
    print("\nType distribution:")
    print(df['y_type'].value_counts(normalize=True))
    print("\nScore distribution:")
    print(df['y_score'].value_counts(normalize=True))

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        type_labels = inputs.pop("type_labels")
        outputs = model(**inputs, labels=labels, type_labels=type_labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    # Unpack predictions and labels
    predictions, labels = eval_pred

    # Separate predictions for score and type
    score_logits, type_logits = predictions
    score_predictions = np.argmax(score_logits, axis=1)
    type_predictions = np.argmax(type_logits, axis=1)

    # Ensure labels are split correctly for score and type
    score_labels, type_labels = labels[0], labels[1]

    return {
        "score_accuracy": accuracy_score(score_labels, score_predictions),
        "score_f1_weighted": f1_score(score_labels, score_predictions, average='weighted'),
        "type_accuracy": accuracy_score(type_labels, type_predictions),
        "type_f1_weighted": f1_score(type_labels, type_predictions, average='weighted'),
    }


if __name__ == '__main__':
    MODEL_NAME = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare datasets
    train_dataset = prepare_dataset('data/Semeval2016/train/train_healines_images_students.csv')
    valid_dataset = prepare_dataset('data/Semeval2016/train/validation_healines_images_students.csv')

    # Initialize multi-task model
    model = RobertaForMultiTaskClassification(MODEL_NAME, num_score_labels=6, num_type_labels=7)

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
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        warmup_ratio=0.1,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="score_f1_weighted",
        greater_is_better=True,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine"
    )

    # Initialize custom trainer
    trainer = MultiTaskTrainer(
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
    model.save_pretrained("./trained_multi_task_roberta1")
    tokenizer.save_pretrained("./trained_multi_task_roberta1")