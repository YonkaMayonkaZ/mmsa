#!/usr/bin/env python
"""
Train multimodal sentiment analysis model
"""
import os
import sys
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import logging.config  # Import the config part of logging
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
# BUG #1 FIX: Added precision_score and recall_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/workspace')
from src.models.attention_fusion import AttentionFusionModel
# Assuming src.data.dataset is defined correctly in your project
# from src.data.dataset import MultimodalDataset
from torch.utils.data import Dataset # Using a generic one for now if the file is not provided.

# A placeholder for your MultimodalDataset if it's not available in this context
class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'id': sample['id'],
            'text': torch.tensor(sample['text'], dtype=torch.float32),
            'visual': torch.tensor(sample['visual'], dtype=torch.float32),
            'audio': torch.tensor(sample['audio'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

from torch.utils.data import DataLoader

# BUG #3 FIX: Removed basicConfig and will load from file in the Trainer
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config_path='/workspace/configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # BUG #3 FIX: Load logging configuration from the YAML file
        with open('/workspace/configs/logging_config.yaml', 'r') as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
        
        self.device = torch.device(self.config['project']['device'])
        self.setup_paths()
        self.setup_tensorboard_logging() # Renamed for clarity
        
    def setup_paths(self):
        """Setup output paths"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.config['paths']['output_dir']) / f'run_{timestamp}'
        self.model_dir = self.run_dir / 'models'
        self.log_dir = self.run_dir / 'logs' # This is for TensorBoard logs
        self.metric_dir = self.run_dir / 'metrics'
        
        for dir_path in [self.model_dir, self.log_dir, self.metric_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_tensorboard_logging(self):
        """Setup tensorboard"""
        self.writer = SummaryWriter(self.log_dir)
        
    def load_data(self):
        """Load preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        data_dir = Path(self.config['paths']['data_dir']) / 'processed'
        
        # Load datasets
        with open(data_dir / 'train.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open(data_dir / 'valid.pkl', 'rb') as f:
            valid_data = pickle.load(f)
        with open(data_dir / 'test.pkl', 'rb') as f:
            test_data = pickle.load(f)
        
        # Create datasets
        train_dataset = MultimodalDataset(train_data)
        valid_dataset = MultimodalDataset(valid_data)
        test_dataset = MultimodalDataset(test_data)
        
        # Create loaders
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
        
        logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    def setup_model(self):
        """Initialize model, optimizer, scheduler"""
        logger.info("Setting up model...")
        
        # Model
        self.model = AttentionFusionModel(self.config).to(self.device)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            text = batch['text'].to(self.device)
            visual = batch['visual'].to(self.device)
            audio = batch['audio'].to(self.device)
            label = batch['label'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(text, visual, audio)
            loss = self.criterion(output, label)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return total_loss / len(self.train_loader), accuracy, f1
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='Validation'):
                text = batch['text'].to(self.device)
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                label = batch['label'].to(self.device)
                
                output = self.model(text, visual, audio)
                
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                labels.extend(label.cpu().numpy())
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

    def test(self):
        """Test model on the test set"""
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                text = batch['text'].to(self.device)
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                label = batch['label'].to(self.device)
                
                output = self.model(text, visual, audio)
                
                predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                labels.extend(label.cpu().numpy())

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.model_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.model_dir / 'best.pt')
            logger.info(f"✓ Saved best model (accuracy: {val_acc:.4f})")
    
    def train(self):
        """Main training loop"""
        logger.info("="*50)
        logger.info("Starting Training")
        logger.info("="*50)
        
        best_val_acc = 0
        patience_counter = 0
        
        metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_acc': [], 'val_f1': []
        }
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validate
            # BUG #2 FIX: Unpack the dictionary returned by validate()
            val_metrics = self.validate()
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
            logger.info(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"  Valid -> Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
            self.writer.add_scalar('F1/train', train_f1, epoch)
            self.writer.add_scalar('F1/validation', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save metrics
            metrics_history['train_loss'].append(train_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['train_f1'].append(train_f1)
            metrics_history['val_acc'].append(val_acc)
            metrics_history['val_f1'].append(val_f1)
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered!")
                break
        
        # Save final metrics
        with open(self.metric_dir / 'training_history.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        logger.info(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        
        # Test on best model
        logger.info("\nEvaluating on test set...")
        checkpoint = torch.load(self.model_dir / 'best.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.test()
        logger.info("\nTest Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        
        # Save test metrics
        with open(self.metric_dir / 'test_metrics.json', 'w') as f:
            # Make confusion matrix JSON serializable
            test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
            json.dump(test_metrics, f, indent=2)
        
        self.writer.close()
        
    def run(self):
        """Run complete training pipeline"""
        self.load_data()
        self.setup_model()
        self.train()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
