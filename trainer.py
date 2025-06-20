import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from config import PhantomHunterConfig
from models import PhantomHunter

class PhantomHunterTrainer:
    """Training pipeline for PhantomHunter"""
    
    def __init__(self, model: PhantomHunter, config: PhantomHunterConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        
    def create_augmented_samples(self, batch_texts: List[str], batch_family_labels: List[int]) -> Tuple[List[str], List[int]]:
        """
        Create augmented samples for contrastive learning
        Simple implementation: return same texts as augmentation for demonstration
        In practice, you might want to implement more sophisticated augmentation
        """
        return batch_texts, batch_family_labels
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch_texts = batch["text"]
            batch_binary_labels = torch.tensor(batch["binary_label"], dtype=torch.long)
            batch_family_labels = torch.tensor(batch["family_label"], dtype=torch.long)
            
            # Create augmented samples for contrastive learning
            aug_texts, aug_family_labels = self.create_augmented_samples(batch_texts, batch["family_label"])
            aug_family_labels = torch.tensor(aug_family_labels, dtype=torch.long)
            
            # Forward pass
            outputs = self.model(
                texts=batch_texts,
                family_labels=batch_family_labels,
                augmented_texts=aug_texts,
                augmented_family_labels=aug_family_labels
            )
            
            # Compute loss
            loss = self.model.compute_loss(
                outputs=outputs,
                binary_labels=batch_binary_labels,
                family_labels=batch_family_labels,
                family_labels_aug=aug_family_labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch_texts = batch["text"]
                batch_binary_labels = torch.tensor(batch["binary_label"], dtype=torch.long)
                
                outputs = self.model(texts=batch_texts)
                predictions = torch.argmax(outputs["detection_logits"], dim=-1)
                
                total_correct += (predictions == batch_binary_labels).sum().item()
                total_samples += len(batch_binary_labels)
        
        accuracy = total_correct / total_samples
        return {"accuracy": accuracy} 