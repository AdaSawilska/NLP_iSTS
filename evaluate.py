import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# Load the trained model and tokenizer
MODEL_DIR = "./trained_sroberta"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Define the preprocessing function (reuse the same one used during training)
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
    return tokenized

def preprocess_test_data(test_file):
    """
    Preprocess test data for evaluation.

    Args:
        test_file (str): Path to the test CSV file.

    Returns:
        Dataset: Tokenized dataset ready for inference.
        pd.DataFrame: Original test data.
    """
    df = pd.read_csv(test_file)
    df["x1"] = df["x1"].fillna("EMPTY").astype(str)
    df["x2"] = df["x2"].fillna("EMPTY").astype(str)
    df["sentence1"] = df["sentence1"].fillna("EMPTY").astype(str)
    df["sentence2"] = df["sentence2"].fillna("EMPTY").astype(str)
    dataset = Dataset.from_pandas(df)

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return processed_dataset, df


def export_to_wa_from_csv(csv_file, wa_output_file):
    # Read the CSV file
    df = csv_file

    # Group data by sentences (sentence1 and sentence2)
    grouped = df.groupby(['sentence1', 'sentence2'])

    with open(wa_output_file, 'w', encoding='utf-8') as wa_file:
        for sentence_idx, ((sentence1, sentence2), group) in enumerate(grouped, start=1):
            # Start a new <sentence> block
            wa_file.write(f'<sentence id="{sentence_idx}" status="">\n')
            wa_file.write(f"// {sentence1}\n")
            wa_file.write(f"// {sentence2}\n")

            # Write the <source> chunks and create a mapping of x1 to word-index pairs
            wa_file.write("<source>\n")
            x1_to_index_map = {}
            index_counter = 1
            for row in group.itertuples():
                if row.x1.strip():  # Only process non-empty x1
                    words_x1 = [word for word in row.x1.split() if word != "EMPTY"]  # Exclude "EMPTY"
                    if words_x1:  # Process only if there are non-empty words
                        x1_to_index_map[row.x1] = [(index_counter + i, word) for i, word in enumerate(words_x1)]
                        for idx, word in x1_to_index_map[row.x1]:
                            wa_file.write(f"{idx} {word} : \n")
                        index_counter += len(words_x1)
            wa_file.write("</source>\n")

            # Write the <translation> chunks and create a mapping of x2 to word-index pairs
            wa_file.write("<translation>\n")
            x2_to_index_map = {}
            index_counter = 1
            for row in group.itertuples():
                if row.x2.strip():  # Only process non-empty x2
                    words_x2 = [word for word in row.x2.split() if word != "EMPTY"]  # Exclude "EMPTY"
                    if words_x2:  # Process only if there are non-empty words
                        x2_to_index_map[row.x2] = [(index_counter + i, word) for i, word in enumerate(words_x2)]
                        for idx, word in x2_to_index_map[row.x2]:
                            wa_file.write(f"{idx} {word} : \n")
                        index_counter += len(words_x2)
            wa_file.write("</translation>\n")

            # Write the <alignment> pairs
            wa_file.write("<alignment>\n")
            for row in group.itertuples():
                # Get the indexes of words in x1 and x2, or use 0 if chunk is empty or only contains "EMPTY"
                if row.x1.strip() and any(word != "EMPTY" for word in row.x1.split()):
                    x1_indexes = " ".join(str(idx) for idx, word in x1_to_index_map.get(row.x1, []))
                else:
                    x1_indexes = "0"

                if row.x2.strip() and any(word != "EMPTY" for word in row.x2.split()):
                    x2_indexes = " ".join(str(idx) for idx, word in x2_to_index_map.get(row.x2, []))
                else:
                    x2_indexes = "0"

                # Validate alignment type (default to EQUI if not provided)
                alignment_type = row.alignment_type if hasattr(row, 'alignment_type') and row.alignment_type in [
                    "EQUI", "OPPO", "SPE1", "SPE2", "SIMI", "REL", "NOALI", "ALIC"] else "EQUI"

                # Validate score (default to 5 or NIL for NOALI/ALIC)
                if alignment_type in {"NOALI", "ALIC"}:
                    score = "NIL"
                else:
                    score = row.predicted_score if hasattr(row, 'predicted_score') and 0 <= row.predicted_score <= 5 else 5

                # Write the alignment line
                wa_file.write(f"{x1_indexes} <==> {x2_indexes} // {alignment_type} // {score} // {row.x1} <==> {row.x2}\n")
            wa_file.write("</alignment>\n")

            # Close the <sentence> block
            wa_file.write("</sentence>\n\n")








# Load and preprocess test data
test_file = "data/Semeval2016/test/test_goldStandard/STSint.testinput.headlines.csv"
test_dataset, test_df = preprocess_test_data(test_file)

# Initialize Trainer
trainer = Trainer(model=model)

# Predict using the model
predictions = trainer.predict(test_dataset)
predicted_scores = np.argmax(predictions.predictions, axis=1)

# Add predictions to the DataFrame
test_df["predicted_score"] = predicted_scores


# Export predictions to .wa file
output_wa_file = "predictions.wa"
export_to_wa_from_csv(test_df, output_wa_file)


