"""
Preprocess and validate CMU-MOSI data using standard splits.
"""
import sys
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_utils import pad_or_truncate, get_data_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='/workspace/configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.max_seq_len = self.config['data']['max_seq_length']
        
    def load_raw_data(self):
        """Load raw downloaded data."""
        cache_file = self.data_dir / 'raw' / 'mosi_raw.pkl'
        if not cache_file.exists():
            logger.error("Raw data not found. Please run 'download_data.py' first!")
            sys.exit(1)
        
        logger.info("Loading raw data from cache...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    def align_features(self, dataset):
        """Align multimodal features to the text modality."""
        logger.info("Aligning features to text timeline...")
        text_feature_name = self.config['data']['features']['text']
        dataset.align(text_feature_name)
        logger.info("✓ Alignment complete")
        return dataset
    
    def process_splits(self, dataset):
        """Process train/valid/test splits using standard SDK folds."""
        logger.info("Processing data splits using standard CMU-SDK folds...")
        
        # IMPROVEMENT: Use standard splits from the SDK
        splits = get_data_splits()
        processed_data = {'train': [], 'valid': [], 'test': []}
        
        all_video_ids = list(dataset[self.config['data']['features']['text']].keys())

        for split_name, video_ids in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            # Find all segments belonging to the videos in this split
            segment_ids = [sid for sid in all_video_ids if sid.split('[')[0] in video_ids]
            
            for segment_id in tqdm(segment_ids, desc=f"  {split_name}"):
                try:
                    text = dataset[self.config['data']['features']['text']][segment_id]['features']
                    visual = dataset[self.config['data']['features']['visual']][segment_id]['features']
                    audio = dataset[self.config['data']['features']['audio']][segment_id]['features']
                    label_score = dataset[self.config['data']['features']['labels']][segment_id]['features'][0][0]
                    
                    text = pad_or_truncate(text, self.max_seq_len)
                    visual = pad_or_truncate(visual, self.max_seq_len)
                    audio = pad_or_truncate(audio, self.max_seq_len)
                    
                    if label_score > 0:
                        label_class = 2  # positive
                    elif label_score < 0:
                        label_class = 0  # negative
                    else:
                        label_class = 1  # neutral
                    
                    processed_data[split_name].append({
                        'id': segment_id,
                        'text': text,
                        'visual': visual,
                        'audio': audio,
                        'label': label_class,
                        'raw_label': label_score
                    })
                except Exception as e:
                    logger.warning(f"Skipping segment {segment_id} due to error: {e}")
                    continue
        
        return processed_data

    def save_processed_data(self, processed_data):
        """Save processed data to pickle files."""
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(exist_ok=True)
        
        for split_name, data in processed_data.items():
            output_file = output_dir / f'{split_name}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"✓ Saved {split_name} data ({len(data)} samples) to {output_file}")

    def compute_statistics(self, processed_data):
        """Compute and log dataset statistics."""
        logger.info("\n--- Dataset Statistics ---")
        for split_name, data in processed_data.items():
            if not data:
                logger.warning(f"{split_name.upper()} split has no data!")
                continue
            labels = [sample['label'] for sample in data]
            total_samples = len(data)
            neg_count = labels.count(0)
            neu_count = labels.count(1)
            pos_count = labels.count(2)
            
            logger.info(f"{split_name.upper()} Split:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Negative (0): {neg_count} ({neg_count/total_samples*100:.1f}%)")
            logger.info(f"  Neutral  (1): {neu_count} ({neu_count/total_samples*100:.1f}%)")
            logger.info(f"  Positive (2): {pos_count} ({pos_count/total_samples*100:.1f}%)")
        logger.info("--------------------------")

    def run(self):
        """Run the complete preprocessing pipeline."""
        logger.info("="*50)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("="*50)
        
        dataset = self.load_raw_data()
        dataset = self.align_features(dataset)
        processed_data = self.process_splits(dataset)
        self.save_processed_data(processed_data)
        self.compute_statistics(processed_data)
        
        logger.info("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
