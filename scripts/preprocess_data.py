#!/usr/bin/env python
"""
Preprocess and validate CMU-MOSI data
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import yaml

sys.path.append('/workspace')
from src.data.data_utils import pad_or_truncate, get_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='/workspace/configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.max_seq_len = self.config['data']['max_seq_length']
        
    def load_raw_data(self):
        """Load raw downloaded data"""
        cache_file = self.data_dir / 'raw' / 'mosi_raw.pkl'
        
        if not cache_file.exists():
            logger.error("Raw data not found. Please run download_data.py first!")
            sys.exit(1)
        
        logger.info("Loading raw data...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        
        return dataset
    
    def align_features(self, dataset):
        """Align multimodal features"""
        logger.info("Aligning features to text timeline...")
        
        # Align all features to text
        dataset.align(self.config['data']['features']['text'])
        
        logger.info("✓ Alignment complete")
        return dataset
    
    def validate_data(self, dataset):
        """Validate data integrity"""
        logger.info("Validating data...")
        
        issues = []
        valid_segments = []
        
        for segment_id in tqdm(dataset.keys(), desc="Validating"):
            try:
                # Check all modalities exist
                text = dataset[self.config['data']['features']['text']][segment_id]
                visual = dataset[self.config['data']['features']['visual']][segment_id]
                audio = dataset[self.config['data']['features']['audio']][segment_id]
                labels = dataset[self.config['data']['features']['labels']][segment_id]
                
                # Check dimensions
                assert text['features'].shape[1] == self.config['data']['dims']['text']
                assert visual['features'].shape[1] == self.config['data']['dims']['visual']
                assert audio['features'].shape[1] == self.config['data']['dims']['audio']
                
                valid_segments.append(segment_id)
                
            except Exception as e:
                issues.append(f"{segment_id}: {str(e)}")
        
        logger.info(f"✓ Valid segments: {len(valid_segments)}/{len(dataset.keys())}")
        
        if issues:
            logger.warning(f"Found {len(issues)} issues:")
            for issue in issues[:5]:  # Show first 5
                logger.warning(f"  - {issue}")
        
        return valid_segments
    
    def process_splits(self, dataset, valid_segments):
        """Process train/valid/test splits"""
        logger.info("Processing data splits...")
        
        splits = {
            'train': self.config['data']['train_split'],
            'valid': self.config['data']['valid_split'],
            'test': self.config['data']['test_split']
        }
        
        processed_data = {}
        
        for split_name, split_ids in splits.items():
            logger.info(f"Processing {split_name} split...")
            split_data = []
            
            for segment_id in tqdm(valid_segments, desc=f"{split_name}"):
                video_id = segment_id.split('[')[0]
                
                if video_id not in split_ids:
                    continue
                
                try:
                    # Extract features
                    text = dataset[self.config['data']['features']['text']][segment_id]['features']
                    visual = dataset[self.config['data']['features']['visual']][segment_id]['features']
                    audio = dataset[self.config['data']['features']['audio']][segment_id]['features']
                    label = dataset[self.config['data']['features']['labels']][segment_id]['features'][0][0]
                    
                    # Pad/truncate to fixed length
                    text = pad_or_truncate(text, self.max_seq_len)
                    visual = pad_or_truncate(visual, self.max_seq_len)
                    audio = pad_or_truncate(audio, self.max_seq_len)
                    
                    # Convert label to classification
                    if label > 0.1:
                        label_class = 2  # positive
                    elif label < -0.1:
                        label_class = 0  # negative
                    else:
                        label_class = 1  # neutral
                    
                    split_data.append({
                        'id': segment_id,
                        'text': text,
                        'visual': visual,
                        'audio': audio,
                        'label': label_class,
                        'raw_label': label
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing {segment_id}: {e}")
                    continue
            
            processed_data[split_name] = split_data
            logger.info(f"  {split_name}: {len(split_data)} samples")
        
        return processed_data
    
    def save_processed_data(self, processed_data):
        """Save processed data"""
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(exist_ok=True)
        
        for split_name, data in processed_data.items():
            output_file = output_dir / f'{split_name}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"✓ Saved {split_name} to {output_file}")
    
    def compute_statistics(self, processed_data):
        """Compute dataset statistics"""
        logger.info("\nDataset Statistics:")
        logger.info("-" * 40)
        
        for split_name, data in processed_data.items():
            labels = [sample['label'] for sample in data]
            label_counts = {i: labels.count(i) for i in range(3)}
            
            logger.info(f"{split_name.upper()} Split:")
            logger.info(f"  Total samples: {len(data)}")
            logger.info(f"  Negative: {label_counts[0]} ({label_counts[0]/len(data)*100:.1f}%)")
            logger.info(f"  Neutral:  {label_counts[1]} ({label_counts[1]/len(data)*100:.1f}%)")
            logger.info(f"  Positive: {label_counts[2]} ({label_counts[2]/len(data)*100:.1f}%)")
    
    def run(self):
        """Run complete preprocessing pipeline"""
        logger.info("="*50)
        logger.info("Data Preprocessing Pipeline")
        logger.info("="*50)
        
        # Load raw data
        dataset = self.load_raw_data()
        
        # Align features
        dataset = self.align_features(dataset)
        
        # Validate
        valid_segments = self.validate_data(dataset)
        
        # Process splits
        processed_data = self.process_splits(dataset, valid_segments)
        
        # Save
        self.save_processed_data(processed_data)
        
        # Statistics
        self.compute_statistics(processed_data)
        
        logger.info("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()