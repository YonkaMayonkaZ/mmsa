"""
Multimodal dataset for CMU-MOSI data
"""
import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal sentiment analysis"""
    
    def __init__(self, data):
        """
        Initialize dataset
        
        Args:
            data (list): List of dictionaries containing:
                - 'text': text features (np.array)
                - 'visual': visual features (np.array) 
                - 'audio': audio features (np.array)
                - 'label': sentiment label (int)
        """
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to torch tensors
        text = torch.FloatTensor(sample['text'])
        visual = torch.FloatTensor(sample['visual'])
        audio = torch.FloatTensor(sample['audio'])
        label = torch.LongTensor([sample['label']])[0]  # Single label
        
        return {
            'text': text,
            'visual': visual,
            'audio': audio,
            'label': label
        }