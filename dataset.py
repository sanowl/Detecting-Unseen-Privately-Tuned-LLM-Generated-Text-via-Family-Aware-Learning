from typing import List
from torch.utils.data import Dataset

class PhantomHunterDataset(Dataset):
    """Dataset for PhantomHunter training"""
    
    def __init__(self, texts: List[str], binary_labels: List[int], family_labels: List[int]):
        self.texts = texts
        self.binary_labels = binary_labels
        self.family_labels = family_labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "binary_label": self.binary_labels[idx],
            "family_label": self.family_labels[idx]
        } 