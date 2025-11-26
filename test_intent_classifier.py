"""
Test the trained intent classifier
"""
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import *


class IntentClassifier:
    """Load and use trained intent classifier"""
    
    def __init__(self, model_path=INTENT_CLASSIFIER_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load intent mappings
        with open(model_path / 'intent_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.id2intent = {int(k): v for k, v in mappings['id2intent'].items()}
        
        print(f"✅ Intent classifier loaded from {model_path}")
    
    def predict(self, text, return_probs=False):
        """Predict intent for a given text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
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


def main():
    """Test the classifier with sample queries"""
    
    print("="*60)
    print("INTENT CLASSIFIER TEST")
    print("="*60)
    
    # Load classifier
    try:
        classifier = IntentClassifier()
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python src/train_intent_classifier.py")
        return
    
    # Test queries
    test_queries = [
        "give me a setup for Monaco",
        "how to drive Silverstone",
        "what is front wing",
        "car is understeering",
        "car is oversteering",
        "tires are overheating",
        "tell me about all tracks",
        "hi there",
        "thanks for the help",
        "brakes are locking up",
        "car feels unstable",
        "I need tips for Spa"
    ]
    
    print("\n🧪 Testing classifier on sample queries:\n")
    
    for query in test_queries:
        intent, confidence = classifier.predict(query)
        
        # Color code based on confidence
        if confidence > 0.9:
            emoji = "✅"
        elif confidence > 0.7:
            emoji = "⚠️"
        else:
            emoji = "❌"
        
        print(f"{emoji} '{query}'")
        print(f"   → Intent: {intent} (confidence: {confidence:.2%})\n")
    
    # Interactive testing
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("Type your queries (or 'quit' to exit):\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input or user_input.lower() == 'quit':
            break
        
        intent, confidence, all_probs = classifier.predict(user_input, return_probs=True)
        
        print(f"\n🎯 Predicted Intent: {intent}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nTop 3 predictions:")
        for i, (intent_name, prob) in enumerate(sorted_probs, 1):
            print(f"  {i}. {intent_name}: {prob:.2%}")
        print()


if __name__ == "__main__":
    main()
