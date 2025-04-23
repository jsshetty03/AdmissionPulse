import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Use PyTorch's native AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class SOPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] - 1  # Convert 1-5 scale to 0-4

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ImprovedSOPRater:
    def __init__(self, model_path="bert-base-uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
    def rate_sop(self, sop_text):
        """Rate a single SOP with detailed analysis"""
        self.model.eval()
    
        inputs = self.tokenizer(
            sop_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
         ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
    
        return {
                 'rating': (top3_indices[0].cpu().numpy() + 1).tolist(),
                 'confidence': top3_probs[0].cpu().numpy().tolist(),
                 'weighted_score': float(torch.sum(probabilities * torch.arange(1, 6).to(self.device)))
    }    
    def train(self, train_data_path, epochs=5, batch_size=8):
        """Train the model on the provided dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(train_data_path)
        
        # Ensure we have samples for all classes
        class_counts = df['rating'].value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            # Duplicate samples for classes with too few instances
            for rating in range(1, 6):
                class_samples = df[df['rating'] == rating]
                if len(class_samples) < 2:
                    df = pd.concat([df, class_samples] * 2, ignore_index=True)

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['sop_text'].values, 
            df['rating'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['rating'].values
        )

        # Create datasets
        train_dataset = SOPDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SOPDataset(val_texts, val_labels, self.tokenizer)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation
            val_accuracy, val_f1 = self._evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Average loss: {total_loss/len(train_loader):.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}, F1 score: {val_f1:.4f}")

    def _evaluate(self, dataloader):
        """Evaluate the model on a dataloader"""
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        return accuracy, f1

    def rate_sop(self, sop_text):
        """Rate a single SOP and provide detailed feedback"""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            sop_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Convert to 1-5 scale
        rating = predicted_class + 1
        
        return {
            'rating': rating,
            'confidence': float(confidence)
        }
