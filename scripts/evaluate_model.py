import sys
import pickle
from pathlib import Path
import logging
import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.attention_fusion import AttentionFusionModel
from src.data.dataset import MultimodalDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Custom collate function to handle batching of pre-padded sequences
    and create an attention mask.
    """
    # Extract data for each modality and labels
    texts = torch.stack([item['text'] for item in batch])
    visuals = torch.stack([item['visual'] for item in batch])
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # The data is already padded/truncated to a fixed length in preprocessing.
    # The mask should identify where the real data is (non-zero padding).
    # We create a mask for the text modality assuming padding is all zeros.
    # A value of `True` in the mask indicates a position that should be ignored.
    attention_mask = (texts.sum(dim=-1) == 0)

    return {
        'text': texts,
        'visual': visuals,
        'audio': audios,
        'label': labels,
        'attention_mask': attention_mask
    }


class Evaluator:
    def __init__(self, model_path, output_dir):
        """
        Initialize the evaluator.
        Args:
            model_path (str): Path to the saved model checkpoint (.pt file).
            output_dir (str): Directory to save the generated plots.
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model_path.is_file():
            logger.error(f"Model checkpoint not found at: {self.model_path}")
            sys.exit(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.load_model()

    def load_model(self):
        """Load the model and configuration from the checkpoint."""
        logger.info(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        self.model = AttentionFusionModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def load_test_data(self):
        """Load the test dataset."""
        data_dir = Path(self.config['paths']['data_dir']) / 'processed'
        test_data_path = data_dir / 'test.pkl'

        if not test_data_path.exists():
            logger.error(f"Test data not found at {test_data_path}. Please run preprocessing first.")
            sys.exit(1)

        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        test_dataset = MultimodalDataset(test_data)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_fn
        )
        logger.info(f"Test data loaded: {len(test_dataset)} samples.")

    def run_inference(self, modality_to_use='all'):
        """
        Run inference on the test set.
        
        Args:
            modality_to_use (str): One of 'all', 'text', 'visual', 'audio'.
                                   Determines which modalities to use for prediction.
                                   Others will be zeroed out.
        
        Returns:
            tuple: A tuple containing lists of true labels and predictions.
        """
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                text = batch['text'].to(self.device)
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # --- EXPERIMENT LOGIC: Zero out unused modalities ---
                if modality_to_use == 'text':
                    visual.zero_()
                    audio.zero_()
                elif modality_to_use == 'visual':
                    text.zero_()
                    audio.zero_()
                elif modality_to_use == 'audio':
                    text.zero_()
                    visual.zero_()

                outputs = self.model(text, visual, audio, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        return all_labels, all_preds

    def plot_confusion_matrix(self, labels, preds, class_names=['Negative', 'Neutral', 'Positive']):
        """Generate and save a confusion matrix plot."""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix for Multimodal Model')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        
        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()

    def plot_performance_comparison(self, results):
        """Generate and save a bar chart comparing model performances."""
        labels = list(results.keys())
        scores = list(results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, scores, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'])
        plt.ylabel('Weighted F1-Score')
        plt.title('Performance Comparison: Multimodal vs. Unimodal')
        plt.ylim(0, 1.0)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

        save_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(save_path, dpi=300)
        logger.info(f"Performance comparison chart saved to {save_path}")
        plt.close()

    def run_all_experiments(self):
        """Execute the full evaluation and experimentation pipeline."""
        self.load_test_data()
        performance_results = {}
        
        modalities_to_test = ['all', 'text', 'visual', 'audio']
        
        for mode in modalities_to_test:
            logger.info(f"\n{'='*20} Evaluating using [{mode.upper()}-ONLY] modality {'='*20}")
            labels, preds = self.run_inference(modality_to_use=mode)
            
            report = classification_report(labels, preds, target_names=['Negative', 'Neutral', 'Positive'], zero_division=0)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            
            performance_results[f'{mode.capitalize()}-Only' if mode != 'all' else 'Multimodal'] = f1
            
            logger.info(f"Classification Report ({mode}):\n{report}")
            
            # Generate the confusion matrix only for the full multimodal model
            if mode == 'all':
                self.plot_confusion_matrix(labels, preds)
        
        # Generate the final comparison plot
        self.plot_performance_comparison(performance_results)
        
        logger.info("\n--- Experiment Summary (Weighted F1-Scores) ---")
        for name, score in performance_results.items():
            logger.info(f"  {name}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal model and run experiments.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint (e.g., outputs/run_TIMESTAMP/models/best.pt)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save the output plots."
    )
    args = parser.parse_args()
    
    evaluator = Evaluator(model_path=args.model_path, output_dir=args.output_dir)
    evaluator.run_all_experiments()

