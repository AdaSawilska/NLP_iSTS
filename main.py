from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust labels

# Define mapping for alignment types
alignment_labels = {"EQUI": 0, "SPE1_FACT": 1, "NOALI": 2}

# Example chunks and alignments
chunks_sentence1 = ["Former Nazi death camp guard Demjanjuk", "dead", "at 91"]
chunks_sentence2 = ["John Demjanjuk", "convicted Nazi death camp guard", "dies aged 91"]
alignments = [
    (["at 91"], ["aged 91"], "EQUI"),
    (["Former Nazi death camp guard Demjanjuk"], ["John Demjanjuk convicted Nazi death camp guard"], "SPE1_FACT"),
    (["dead"], ["dies"], "EQUI")
]

# Prepare inputs and labels
inputs = []
labels = []
for chunk1, chunk2, label in alignments:
    pair = f"[CLS] {chunk1[0]} [SEP] {chunk2[0]} [SEP]"
    tokenized = tokenizer(pair, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    inputs.append(tokenized)
    labels.append(alignment_labels[label])

# Convert to tensors
input_ids = torch.cat([i["input_ids"] for i in inputs])
attention_masks = torch.cat([i["attention_mask"] for i in inputs])
segment_ids = torch.cat([i["token_type_ids"] for i in inputs])
labels = torch.tensor(labels)

# Model input
outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=segment_ids, labels=labels)
loss, logits = outputs.loss, outputs.logits

# Prediction
predicted_labels = torch.argmax(logits, dim=1)