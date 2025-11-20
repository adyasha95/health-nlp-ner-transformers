import json
import random

symptoms = ["fever", "nausea", "chest pain", "dizziness", "shortness of breath"]
medications = ["metformin", "amoxicillin", "ibuprofen", "atorvastatin"]
measurements = ["BP 140/90", "HR 110", "SpO2 93%", "Temp 38.5C"]
conditions = ["suspected infection", "stable condition", "follow-up required"]

def create_example():
    template = f"Patient reports {random.choice(symptoms)}. Started on {random.choice(medications)}. {random.choice(measurements)} noted. Assessment: {random.choice(conditions)}."
    entities = []

    text = template

    def add_entity(word, label):
        start = text.index(word)
        end = start + len(word)
        entities.append({"start": start, "end": end, "label": label})

    add_entity(random.choice(symptoms), "SYMPTOM")
    add_entity(random.choice(medications), "MEDICATION")
    add_entity(random.choice(measurements), "MEASUREMENT")
    add_entity(random.choice(conditions), "CONDITION")

    return {"text": text, "entities": entities}

data = [create_example() for _ in range(500)]

with open("data/synthetic_ner_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Synthetic NER dataset created!")
