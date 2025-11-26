"""
Train Intent Classifier using DistilBERT
Optimized for RTX 3050 4GB VRAM
"""
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from config import *


class IntentClassifierTrainer:
    """Train and evaluate DistilBERT for intent classification"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Load training data
        with open(PROCESSED_DATA_DIR / "intent_training_data.json", 'r') as f:
            data = json.load(f)
            self.training_data = data['training_data']
        
        # Create intent to ID mapping
        self.intents = INTENTS
        self.intent2id = {intent: idx for idx, intent in enumerate(self.intents)}
        self.id2intent = {idx: intent for intent, idx in self.intent2id.items()}
        
        print(f"📊 Loaded {len(self.training_data)} training examples")
        print(f"🎯 Number of intents: {len(self.intents)}")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
    def prepare_dataset(self):
        """Prepare train/validation split"""
        
        # Convert to format for datasets
        texts = [item['text'] for item in self.training_data]
        labels = [self.intent2id[item['intent']] for item in self.training_data]
        
        # Split train/val (80/20)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"✂️  Train size: {len(train_texts)}, Val size: {len(val_texts)}")
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'label': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'label': val_labels
        })
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        self.train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
    def train(self):
        """Train the model"""
        
        print("\n🚀 Starting training...")
        
        # Initialize model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(self.intents)
        )
        model.to(self.device)
        
        # Training arguments (optimized for 4GB VRAM)
        training_args = TrainingArguments(
            output_dir=str(INTENT_CLASSIFIER_DIR),
            num_train_epochs=10,
            per_device_train_batch_size=8,  # Small batch for 4GB VRAM
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=str(OUTPUTS_DIR / 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,  # Keep only 2 best checkpoints
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            gradient_accumulation_steps=2,  # Simulate larger batch
            learning_rate=2e-5
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            
            return {'accuracy': accuracy}
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        print("\n💾 Saving model...")
        trainer.save_model(str(INTENT_CLASSIFIER_DIR))
        self.tokenizer.save_pretrained(str(INTENT_CLASSIFIER_DIR))
        
        # Save intent mappings
        with open(INTENT_CLASSIFIER_DIR / 'intent_mappings.json', 'w') as f:
            json.dump({
                'intent2id': self.intent2id,
                'id2intent': self.id2intent
            }, f, indent=2)
        
        print(f"✅ Model saved to {INTENT_CLASSIFIER_DIR}")
        
        return trainer
    
    def evaluate(self, trainer):
        """Evaluate on validation set"""
        
        print("\n📊 Evaluating model...")
        
        # Get predictions
        predictions = trainer.predict(self.val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            labels,
            preds,
            target_names=[self.id2intent[i] for i in range(len(self.intents))],
            zero_division=0
        )
        print(report)
        
        # Overall accuracy
        accuracy = accuracy_score(labels, preds)
        print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")
        
        return accuracy


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("F1 CHATBOT - INTENT CLASSIFIER TRAINING")
    print("="*60)
    
    # Create output directories
    INTENT_CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer_obj = IntentClassifierTrainer()
    
    # Prepare data
    trainer_obj.prepare_dataset()
    
    # Train
    trainer = trainer_obj.train()
    
    # Evaluate
    accuracy = trainer_obj.evaluate(trainer)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Final Accuracy: {accuracy:.2%}")
    print(f"💾 Model saved to: {INTENT_CLASSIFIER_DIR}")
    print("\nNext steps:")
    print("1. Test the classifier with: python test_intent_classifier.py")
    print("2. Integrate into chatbot: Update src/chatbot.py to use trained model")
    print("="*60)


if __name__ == "__main__":
    main()
