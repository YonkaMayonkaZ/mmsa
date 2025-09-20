import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading multimodal data.
    """
    def __init__(self, data):
        """
        Args:
            data (list): A list of dictionaries, where each dictionary represents a data sample.
        """
        self.data = data

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed multimodal features and label as PyTorch tensors.
        """
        sample = self.data[idx]
        
        # Convert numpy arrays to PyTorch tensors
        return {
            'id': sample['id'],
            'text': torch.tensor(sample['text'], dtype=torch.float32),
            'visual': torch.tensor(sample['visual'], dtype=torch.float32),
            'audio': torch.tensor(sample['audio'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }
