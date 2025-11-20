import json
import argparse
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report, f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-cased")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

with open("data/synthetic_ner_data.json") as f:
    raw = json.load(f)

tokens = []
labels = []

for example in raw:
    text = example["text"].split()
    token_labels = ["O"] * len(text)

    for ent in example["entities"]:
        span = example["text"][ent["start"]:ent["end"]]
        for i, word in enumerate(text):
            if span in word:
                token_labels[i] = ent["label"]

    tokens.append(text)
    labels.append(token_labels)

dataset = Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

label_list = sorted({tag for seq in labels for tag in seq})
label_to_id = {label: i for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

def encode(batch):
    encodings = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True, padding=True)
    encoded_labels = []
    for i, label_seq in enumerate(batch["ner_tags"]):
        word_ids = encodings.word_ids(batch_index=i)
        encoded_labels.append([label_to_id[label_seq[word]] if word is not None else -100 for word in word_ids])
    encodings["labels"] = encoded_labels
    return encodings

dataset = dataset.map(encode, batched=True)
dataset.set_format("torch")

model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))

training_args = TrainingArguments(
    output_dir="models",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    load_best_model_at_end=True,
)

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=-1)
    true = p.label_ids
    pred_labels = [[label_list[p] for p, l in zip(pred_row, true_row) if l != -100]
                   for pred_row, true_row in zip(predictions, true)]
    true_labels = [[label_list[l] for p, l in zip(pred_row, true_row) if l != -100]
                   for pred_row, true_row in zip(predictions, true)]
    return {"f1": f1_score(true_labels, pred_labels)}

trainer = Trainer(model=model, args=training_args, train_dataset=dataset, eval_dataset=dataset, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("models/best_ner_model")
