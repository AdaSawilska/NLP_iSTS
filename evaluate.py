from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, AutoConfig
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from main import RobertaForMultiTaskClassification
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaForMultiTaskClassification(nn.Module):
    def __init__(self, config, num_score_labels=6, num_type_labels=8):
        super().__init__()
        self.roberta = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.score_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_score_labels)
        )
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_type_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, type_labels=None, return_dict=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state[:, 0, :]
        score_logits = self.score_classifier(sequence_output)
        type_logits = self.type_classifier(sequence_output)

        # Pack both logits together as a tuple
        combined_logits = (score_logits, type_logits)

        return SequenceClassifierOutput(
            loss=None,
            logits=combined_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_trained_model(model_dir):
    # Load the configuration
    config = AutoConfig.from_pretrained(model_dir)

    # Create model with the loaded config
    model = RobertaForMultiTaskClassification(config)

    # Load the classification heads
    classification_heads = torch.load(
        f"{model_dir}/classification_heads.pt",
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    # Load the weights
    model.score_classifier.load_state_dict(classification_heads['score_classifier'])
    model.type_classifier.load_state_dict(classification_heads['type_classifier'])

    # Set to evaluation mode
    model.eval()

    return model

class MultiTaskTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            score_logits, type_logits = outputs.logits

        return (None, (score_logits, type_logits), None)




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



if __name__ == "__main__":
    # Modified evaluation code
    MODEL_DIR = "./trained_multi_task_roberta"
    model = load_trained_model(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Use the model for predictions
    test_file = "data/Semeval2016/test/test_goldStandard/STSint.testinput.headlines.csv"
    test_dataset, test_df = preprocess_test_data(test_file)

    trainer = MultiTaskTrainer(model=model)
    predictions = trainer.predict(test_dataset)
    score_logits, type_logits = predictions.predictions
    predicted_scores = np.argmax(score_logits, axis=1)
    predicted_types = np.argmax(type_logits, axis=1)

    # Add predictions to the DataFrame
    test_df["predicted_score"] = predicted_scores
    test_df["predicted_type"] = predicted_types

    # Export predictions to .wa file
    output_wa_file = "predictions.wa"
    export_to_wa_from_csv(test_df, output_wa_file)