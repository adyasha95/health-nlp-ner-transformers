import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from seqeval.metrics import classification_report

model_path = "models/best_ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

with open("data/synthetic_ner_data.json") as f:
    raw = json.load(f)

texts = [x["text"] for x in raw]

def predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logits = model(**tokens).logits
    ids = logits.argmax(-1)[0]
    return ids

print("NER evaluation script ready!")
