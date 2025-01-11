from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from typing import Tuple

from calculate_metrics import prepare_dataframe, calculate_metrics
from main import RobertaForMultiTaskClassification
from predictions_to_wa import create_wa_file_from_predictions


def load_trained_model(model_path):
    """Load the trained multi-task model"""
    model = RobertaForMultiTaskClassification("roberta-base",
                                              num_score_labels=6,
                                              num_type_labels=7)

    state_dict = torch.load(f"{model_path}/model.pt")
    model.load_state_dict(state_dict, strict=True)

    # Set to evaluation mode
    model.eval()
    return model

def validate_model_loading(model_path):
    """Sanity check model loading by verifying weights are loaded."""
    model = load_trained_model(model_path)
    for name, param in model.named_parameters():
        if param.requires_grad and param.sum().item() == 0:
            print(f"Warning: Parameter {name} seems to be uninitialized or zeroed out.")
        if torch.sum(param.data) == 0:
            print(f"Parameter {name} is still uninitialized.")

    return model


def preprocess_function(examples, tokenizer):
    """Preprocess the data for the model"""
    texts = [
        f"{str(x1)} </s></s> {str(x2)} </s></s> {str(s1)} </s></s> {str(s2)}"
        for x1, x2, s1, s2 in zip(
            examples['x1'],
            examples['x2'],
            examples['sentence1'],
            examples['sentence2']
        )
    ]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )


def preprocess_test_data(test_file: str, tokenizer):
    """Load test data and preprocess"""
    try:
        df = pd.read_csv(test_file)
    except:
        df = pd.read_csv(test_file, sep=',')

    df["x1"] = df["x1"].fillna("EMPTY").astype(str)
    df["x2"] = df["x2"].fillna("EMPTY").astype(str)
    df["sentence1"] = df["sentence1"].fillna("EMPTY").astype(str)
    df["sentence2"] = df["sentence2"].fillna("EMPTY").astype(str)

    dataset = Dataset.from_pandas(df)
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["y_score", "y_type"] + dataset.column_names
    )

    return processed_dataset, df


def get_predictions(model, dataset) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collator)

    all_score_preds = []
    all_type_preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**inputs)
            score_logits, type_logits = outputs.logits

            score_preds = torch.argmax(score_logits, dim=1).cpu().numpy()
            type_preds = torch.argmax(type_logits, dim=1).cpu().numpy()

            all_score_preds.extend(score_preds)
            all_type_preds.extend(type_preds)

    return np.array(all_score_preds), np.array(all_type_preds)


if __name__ == "__main__":
    # Variables to adjust
    MODEL_PATH = "./trained_multi_task_roberta3"
    GT = "data/Semeval2016/test/test_goldStandard/STSint.testinput.images.wa"
    TEST_FILE = "data/Semeval2016/test/test_goldStandard/STSint.testinput.images.csv"
    OUTPUT_WA = 'predictions_test_images.wa'


    # Load model and tokenizer
    model = validate_model_loading(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(model)
    print(tokenizer)

    # Load and preprocess test data
    test_dataset, test_df = preprocess_test_data(TEST_FILE, tokenizer)

    # Get predictions
    predicted_scores, predicted_types = get_predictions(model, test_dataset)

    # Add predictions to DataFrame
    test_df["predicted_score"] = predicted_scores
    test_df["predicted_type"] = predicted_types

    # Save to .csv
    test_df.to_csv('predictions_test_images.csv', index=False)

    # Save to .wa
    create_wa_file_from_predictions(test_df, GT, OUTPUT_WA)

    # Calculate metrics
    df_prepared = prepare_dataframe(test_df)
    calculate_metrics(df_prepared)

