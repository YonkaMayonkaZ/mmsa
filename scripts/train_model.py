#!/usr/bin/env python
"""
Train multimodal sentiment analysis model
"""
import sys
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import logging.config
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.attention_fusion import AttentionFusionModel
from src.data.dataset import MultimodalDataset

# Set up logger. The configuration will be applied in the Trainer.
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path='/workspace/configs/config.yaml'):
        # Load main configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load and apply logging configuration
        with open('/workspace/configs/logging_config.yaml', 'r') as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
        
        self.device = torch.device(self.config['project']['device'] if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.setup_paths()
        self.setup_tensorboard()
        
    def setup_paths(self):
        """Setup dynamic output paths for the current run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.config['paths']['output_dir']) / f"run_{timestamp}"
        self.model_dir = self.run_dir / 'models'
        self.log_dir = self.run_dir / 'logs'
        self.metric_dir = self.run_dir / 'metrics'
        
        for dir_path in [self.model_dir, self.log_dir, self.metric_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_tensorboard(self):
        """Initialize TensorBoard SummaryWriter."""
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def load_data(self):
        """Load preprocessed data into PyTorch DataLoaders."""
        logger.info("Loading preprocessed data...")
        data_dir = Path(self.config['paths']['data_dir']) / 'processed'
        
        with open(data_dir / 'train.pkl', 'rb') as f: train_data = pickle.load(f)
        with open(data_dir / 'valid.pkl', 'rb') as f: valid_data = pickle.load(f)
        with open(data_dir / 'test.pkl', 'rb') as f: test_data = pickle.load(f)
        
        train_dataset = MultimodalDataset(train_data)
        valid_dataset = MultimodalDataset(valid_data)
        test_dataset = MultimodalDataset(test_data)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=self.config['data']['num_workers'])
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=self.config['data']['num_workers'])
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=self.config['data']['num_workers'])
        
        logger.info(f"Data loaded. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)} samples.")
    
    def setup_model(self):
        """Initialize model, loss function, optimizer, and scheduler."""
        logger.info("Setting up model...")
        self.model = AttentionFusionModel(self.config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['training']['learning_rate'], weight_decay=self.config['training']['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['training']['epochs'])
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss, predictions, labels = 0, [], []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            text, visual, audio, label = batch['text'].to(self.device), batch['visual'].to(self.device), batch['audio'].to(self.device), batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(text, visual, audio)
            loss = self.criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return avg_loss, acc, f1
    
    def evaluate(self, loader, description='Eval'):
        """Evaluate the model on a given data loader."""
        self.model.eval()
        predictions, labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=description):
                text, visual, audio, label = batch['text'].to(self.device), batch['visual'].to(self.device), batch['audio'].to(self.device), batch['label'].to(self.device)
                output = self.model(text, visual, audio)
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                labels.extend(label.cpu().numpy())
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions)
        }
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc
        }
        if is_best:
            torch.save(checkpoint, self.model_dir / 'best.pt')
            logger.info(f"✓ Saved new best model (Val Acc: {val_acc:.4f})")
    
    def train(self):
        """Main training loop."""
        logger.info("="*50)
        logger.info("Starting Model Training")
        logger.info("="*50)
        
        best_val_acc, patience_counter = 0, 0
        history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.valid_loader, description=f'Epoch {epoch} [Valid]')
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch}/{self.config['training']['epochs']} | Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Valid -> Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/valid', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/valid', val_metrics['f1'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # History
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc); history['train_f1'].append(train_f1)
            history['val_acc'].append(val_metrics['accuracy']); history['val_f1'].append(val_metrics['f1'])
            
            # Checkpointing & Early Stopping
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_checkpoint(epoch, best_val_acc, is_best=True)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered!")
                break
        
        # Final evaluation on the test set with the best model
        logger.info("\nTraining complete. Evaluating on test set with best model...")
        checkpoint = torch.load(self.model_dir / 'best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(self.test_loader, description='Testing')
        
        logger.info("\n--- Test Results ---")
        for key, value in test_metrics.items():
            if key != 'confusion_matrix':
                logger.info(f"  {key.capitalize()}: {value:.4f}")
        logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
        # Save metrics
        test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        with open(self.metric_dir / 'test_metrics.json', 'w') as f: json.dump(test_metrics, f, indent=2)
        with open(self.metric_dir / 'training_history.json', 'w') as f: json.dump(history, f, indent=2)
        
        self.writer.close()

    def run(self):
        """Run the complete training pipeline."""
        self.load_data()
        self.setup_model()
        self.train()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

