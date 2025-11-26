"""
IntentClassifier wrapper used by chatbot and test script
Loads the trained DistilBERT intent classifier
"""

import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "intent_classifier"


class IntentClassifier:
    """Load and use trained intent classifier"""

    def __init__(self, model_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = MODEL_DIR
        self.model_path = Path(model_path)

        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load intent mappings
        with open(self.model_path / "intent_mappings.json", "r") as f:
            mappings = json.load(f)
            # keys in JSON are strings, convert to int
            self.id2intent = {int(k): v for k, v in mappings["id2intent"].items()}

        print(f"✅ Intent classifier loaded from {self.model_path}")

    def predict(self, text: str, return_probs: bool = False):
        """Predict intent for a given text"""

        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        intent = self.id2intent[predicted_class]

        if return_probs:
            all_probs = {self.id2intent[i]: probs[0][i].item() for i in range(len(self.id2intent))}
            return intent, confidence, all_probs

        return intent, confidence
