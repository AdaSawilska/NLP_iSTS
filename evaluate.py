from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch
from typing import Tuple
from main import RobertaForMultiTaskClassification  # Import from your main module


def load_trained_model(model_path: str) -> RobertaForMultiTaskClassification:
    """Load the trained multi-task model"""
    model = RobertaForMultiTaskClassification("roberta-base", num_score_labels=6, num_type_labels=7)

    # Load the saved weights
    checkpoint = torch.load(
        f"{model_path}/classification_heads.pt",
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    model.score_projection.load_state_dict(checkpoint['score_projection'])
    model.type_projection.load_state_dict(checkpoint['type_projection'])
    model.score_classifier.load_state_dict(checkpoint['score_classifier'])
    model.type_classifier.load_state_dict(checkpoint['type_classifier'])
    model.log_vars = checkpoint['log_vars']

    # Set to evaluation mode
    model.eval()
    return model

def validate_model_loading(model_path: str):
    """Sanity check model loading by verifying weights are loaded."""
    model = load_trained_model(model_path)
    for name, param in model.named_parameters():
        if param.requires_grad and param.sum().item() == 0:
            print(f"Warning: Parameter {name} seems to be uninitialized or zeroed out.")
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


def preprocess_test_data(test_file: str, tokenizer) -> Tuple[Dataset, pd.DataFrame]:
    """Preprocess test data for evaluation"""
    try:
        df = pd.read_csv(test_file)
    except:
        # Try with different separator if default fails
        df = pd.read_csv(test_file, sep='\t')

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
    """Get predictions from the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use a data collator to handle padding dynamically
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


def export_to_wa_format(df: pd.DataFrame, output_file: str):
    """Export predictions to WA format"""
    type_mapping = {0: 'EQUI', 1: 'OPPO', 2: 'SPE1', 3: 'SPE2',
                    4: 'SIMI', 5: 'REL', 6: 'NOALI', 7: 'ALIC'}

    df['alignment_type'] = df['predicted_type'].map(type_mapping)

    with open(output_file, 'w', encoding='utf-8') as wa_file:
        for sentence_idx, ((sentence1, sentence2), group) in enumerate(
                df.groupby(['sentence1', 'sentence2']), start=1
        ):
            wa_file.write(f'<sentence id="{sentence_idx}" status="">\n')
            wa_file.write(f"// {sentence1}\n")
            wa_file.write(f"// {sentence2}\n")

            # Write source
            wa_file.write("<source>\n")
            source_words = {}
            idx = 1
            for row in group.itertuples():
                if row.x1.strip() and row.x1 != "EMPTY":
                    words = row.x1.split()
                    source_words[row.x1] = list(range(idx, idx + len(words)))
                    for word_idx, word in enumerate(words, idx):
                        wa_file.write(f"{word_idx} {word} : \n")
                    idx += len(words)
            wa_file.write("</source>\n")

            # Write translation
            wa_file.write("<translation>\n")
            target_words = {}
            idx = 1
            for row in group.itertuples():
                if row.x2.strip() and row.x2 != "EMPTY":
                    words = row.x2.split()
                    target_words[row.x2] = list(range(idx, idx + len(words)))
                    for word_idx, word in enumerate(words, idx):
                        wa_file.write(f"{word_idx} {word} : \n")
                    idx += len(words)
            wa_file.write("</translation>\n")

            # Write alignments
            wa_file.write("<alignment>\n")
            for row in group.itertuples():
                source_indices = " ".join(map(str, source_words.get(row.x1, ["0"])))
                target_indices = " ".join(map(str, target_words.get(row.x2, ["0"])))
                score = "NIL" if row.alignment_type in ["NOALI", "ALIC"] else str(row.predicted_score)
                wa_file.write(
                    f"{source_indices} <==> {target_indices} // {row.alignment_type} // {score} // {row.x1} <==> {row.x2}\n")
            wa_file.write("</alignment>\n")

            wa_file.write("</sentence>\n\n")


if __name__ == "__main__":
    # Load model and tokenizer
    MODEL_PATH = "./trained_multi_task_roberta1"  # Update with your model path
    model = validate_model_loading(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load and preprocess test data
    test_file = "data/Semeval2016/train/train_healines_images_students.csv"  # Update with your test file path
    test_dataset, test_df = preprocess_test_data(test_file, tokenizer)

    # Get predictions
    predicted_scores, predicted_types = get_predictions(model, test_dataset)

    # Add predictions to DataFrame
    test_df["predicted_score"] = predicted_scores
    test_df["predicted_type"] = predicted_types

    # # Export to WA format
    # output_wa_file = "predictions.wa"
    # export_to_wa_format(test_df, output_wa_file)
    #
    # # Print some sample predictions
    # print("\nSample predictions:")
    # for i in range(min(5, len(test_df))):
    #     print(f"\nPrediction {i + 1}:")
    #     print(f"Sentence 1: {test_df.iloc[i]['sentence1']}")
    #     print(f"Sentence 2: {test_df.iloc[i]['sentence2']}")
    #     print(f"Predicted Score: {predicted_scores[i]}")
    #     print(f"Predicted Type: {predicted_types[i]}")