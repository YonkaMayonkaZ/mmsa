"""
Preprocess CMU-MOSI data directly from an HDF5 file.
"""
import sys
import pickle
from pathlib import Path
import logging
import h5py
import numpy as np
from tqdm import tqdm
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_utils import pad_or_truncate, get_data_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.max_seq_len = self.config['data']['max_seq_length']
        # Path to your HDF5 file
        self.hdf5_path = self.data_dir / 'raw' / 'mosi.hdf5'

    def load_from_hdf5(self):
        """Load data directly from the CMU-MOSI HDF5 file."""
        processed_data = {'train': [], 'valid': [], 'test': []}
        
        logger.info(f"Loading data from HDF5 file: {self.hdf5_path}")
        if not self.hdf5_path.exists():
            logger.error(f"HDF5 file not found at {self.hdf5_path}. Please place 'mosi.hdf5' in the 'data/raw/' directory.")
            sys.exit(1)

        # Use the standard data splits
        splits = get_data_splits()
        
        with h5py.File(self.hdf5_path, 'r') as hf:
            # Get feature names from config
            audio_feature = self.config['data']['features']['audio']
            visual_feature = self.config['data']['features']['visual']
            text_feature = self.config['data']['features']['text']
            label_feature = self.config['data']['features']['labels']
            
            all_segment_ids = list(hf[label_feature].keys())

            for split_name, video_ids in splits.items():
                logger.info(f"Processing {split_name} split...")
                
                # Find all segments belonging to the videos in this split
                segment_ids_for_split = [sid for sid in all_segment_ids if sid.split('[')[0] in video_ids]

                for segment_id in tqdm(segment_ids_for_split, desc=f"  {split_name}"):
                    try:
                        # NOTE: The HDF5 'words' are not GloVe vectors. 
                        # A real pipeline would convert words to vectors here.
                        # For now, we create placeholder 'text' features to match expected dimensions.
                        num_words = hf[text_feature][segment_id]['features'].shape[0]
                        text_dim = self.config['data']['dims']['text']
                        text = np.random.rand(num_words, text_dim) # Placeholder

                        visual = hf[visual_feature][segment_id]['features'][()]
                        audio = hf[audio_feature][segment_id]['features'][()]
                        label_score = hf[label_feature][segment_id]['features'][()][0][0]

                        # Pad or truncate features
                        text = pad_or_truncate(text, self.max_seq_len)
                        visual = pad_or_truncate(visual, self.max_seq_len)
                        audio = pad_or_truncate(audio, self.max_seq_len)
                        
                        # Convert label score to class
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
                    except KeyError as e:
                        logger.warning(f"Skipping segment {segment_id} due to missing key: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Skipping segment {segment_id} due to an error: {e}")
                        continue
        
        return processed_data

    def save_processed_data(self, processed_data):
        """Save processed data to pickle files."""
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        logger.info("Starting Data Preprocessing Pipeline from HDF5")
        logger.info("="*50)
        
        processed_data = self.load_from_hdf5()
        self.save_processed_data(processed_data)
        self.compute_statistics(processed_data)
        
        logger.info("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    # Ensure the script is run from the root of the project
    # so the config paths are correct.
    preprocessor = DataPreprocessor(config_path='configs/config.yaml')
    preprocessor.run()