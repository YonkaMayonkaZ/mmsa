#!/usr/bin/env python
"""
Train multimodal sentiment analysis model - Simple version with bidirectional attention
"""
import sys
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config['project']['seed'])
        np.random.seed(self.config['project']['seed'])
        
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
        
        # Check if processed data exists
        for split in ['train', 'valid', 'test']:
            file_path = data_dir / f'{split}.pkl'
            if not file_path.exists():
                logger.error(f"Processed data not found: {file_path}")
                logger.error("Please run preprocessing first: python scripts/preprocess_data.py")
                sys.exit(1)
        
        with open(data_dir / 'train.pkl', 'rb') as f: 
            train_data = pickle.load(f)
        with open(data_dir / 'valid.pkl', 'rb') as f: 
            valid_data = pickle.load(f)
        with open(data_dir / 'test.pkl', 'rb') as f: 
            test_data = pickle.load(f)
        
        train_dataset = MultimodalDataset(train_data)
        valid_dataset = MultimodalDataset(valid_data)
        test_dataset = MultimodalDataset(test_data)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=True, 
            num_workers=self.config['data']['num_workers']
        )
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=False, 
            num_workers=self.config['data']['num_workers']
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=False, 
            num_workers=self.config['data']['num_workers']
        )
        
        logger.info(f"Data loaded. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)} samples.")
    
    def setup_model(self):
        """Initialize model, loss function, optimizer, and scheduler."""
        logger.info("Setting up bidirectional attention model...")
        self.model = AttentionFusionModel(self.config).to(self.device)
        
        # Use label smoothing to prevent overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'], 
            weight_decay=self.config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs']
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            text = batch['text'].to(self.device)
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(text, visual, audio)
            loss = self.criterion(outputs, labels)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }
    
    def evaluate(self, data_loader, description="Validation"):
        """Evaluate the model on validation/test data."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=description):
                text = batch['text'].to(self.device)
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(text, visual, audio)
                loss = self.criterion(outputs, labels)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'confusion_matrix': cm
        }

    def train(self):
        """Main training loop."""
        logger.info("Starting training with bidirectional attention...")
        
        best_val_f1 = 0.0
        patience_counter = 0
        history = {'train': [], 'valid': []}
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            logger.info(f"\n--- Epoch {epoch}/{self.config['training']['epochs']} ---")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.valid_loader, "Validation")
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            logger.info(f"Valid - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save history
            history['train'].append(train_metrics)
            history['valid'].append({k: v for k, v in val_metrics.items() if k != 'confusion_matrix'})
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_f1': val_metrics['f1'],
                    'config': self.config
                }, self.model_dir / 'best.pt')
                
                logger.info(f"✓ New best model saved! F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        checkpoint = torch.load(self.model_dir / 'best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics = self.evaluate(self.test_loader, description='Testing')
        
        logger.info("\n--- Test Results ---")
        for key, value in test_metrics.items():
            if key != 'confusion_matrix':
                logger.info(f"  {key.capitalize()}: {value:.4f}")
        logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
        
        # Save results
        test_results = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
        test_results['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        
        with open(self.metric_dir / 'test_metrics.json', 'w') as f: 
            json.dump(test_results, f, indent=2)
        with open(self.metric_dir / 'training_history.json', 'w') as f: 
            json.dump(history, f, indent=2)
        
        self.writer.close()

    def run(self):
        """Run the complete training pipeline."""
        self.load_data()
        self.setup_model()
        self.train()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()