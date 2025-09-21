"""
Preprocess CMU-MOSI data with BERT text embeddings
"""
import sys
import pickle
from pathlib import Path
import logging
import h5py
import numpy as np
from tqdm import tqdm
import yaml
import torch
from transformers import AutoTokenizer, AutoModel

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_utils import pad_or_truncate, get_data_splits_from_segments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.max_seq_len = self.config['data']['max_seq_length']
        self.hdf5_path = self.data_dir / 'raw' / 'mosi.hdf5'
        
        # Initialize BERT model
        self.setup_bert()

    def setup_bert(self):
        """Initialize BERT model for text embeddings"""
        logger.info("Loading BERT model for text embeddings...")
        
        # Use a smaller BERT model for efficiency
        model_name = "distilbert-base-uncased"  # Smaller and faster than full BERT
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # Set to evaluation mode and move to GPU if available
            self.bert_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            
            logger.info(f"BERT model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            logger.info("Falling back to simple text features...")
            self.bert_model = None
            self.tokenizer = None

    def get_bert_embeddings(self, words):
        """Get BERT embeddings for a list of words"""
        if self.bert_model is None or not words:
            # Fallback to zeros if BERT not available
            return np.zeros(self.config['data']['dims']['text'], dtype=np.float32)
        
        try:
            # Join words into text
            text = ' '.join(words)
            
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            # Resize to match expected dimensions
            expected_dim = self.config['data']['dims']['text']
            if len(cls_embedding) > expected_dim:
                # Truncate if BERT embedding is larger
                embedding = cls_embedding[:expected_dim]
            else:
                # Pad if BERT embedding is smaller
                embedding = np.zeros(expected_dim, dtype=np.float32)
                embedding[:len(cls_embedding)] = cls_embedding
            
            # Normalize to prevent large values
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"BERT embedding failed: {e}, using fallback")
            return np.zeros(self.config['data']['dims']['text'], dtype=np.float32)

    def create_simple_text_features(self, words):
        """Fallback simple text features if BERT fails"""
        text_dim = self.config['data']['dims']['text']
        clean_words = [w for w in words if w and w != 'sp' and len(w.strip()) > 0]
        
        if not clean_words:
            return np.zeros(text_dim, dtype=np.float32)
        
        features = np.zeros(text_dim, dtype=np.float32)
        
        # Basic statistics
        features[0] = min(len(clean_words) / 20.0, 1.0)
        features[1] = min(np.mean([len(w) for w in clean_words]) / 10.0, 1.0)
        features[2] = min(len(set(clean_words)) / len(clean_words), 1.0)
        
        # Simple word hashes
        for i, word in enumerate(clean_words[:10]):
            word_hash = hash(word.lower()) % 1000
            if 3 + i < text_dim:
                features[3 + i] = word_hash / 1000.0
        
        return features.astype(np.float32)

    def load_from_hdf5(self):
        """Load data using real features with BERT text embeddings."""
        processed_data = {'train': [], 'valid': [], 'test': []}
        
        logger.info(f"Loading data from HDF5 file: {self.hdf5_path}")
        if not self.hdf5_path.exists():
            logger.error(f"HDF5 file not found at {self.hdf5_path}.")
            sys.exit(1)

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_feature = self.config['data']['features']['audio']
            visual_feature = self.config['data']['features']['visual']
            text_feature = self.config['data']['features']['text']
            label_feature = self.config['data']['features']['labels']
            
            all_segment_ids = list(hf[label_feature].keys())
            logger.info(f"Found {len(all_segment_ids)} segments in dataset")

            splits = get_data_splits_from_segments(all_segment_ids)

            for split_name, video_ids in splits.items():
                logger.info(f"Processing {split_name} split...")
                
                segment_ids_for_split = [sid for sid in all_segment_ids if sid.split('[')[0] in video_ids]
                logger.info(f"  Found {len(segment_ids_for_split)} segments for {split_name}")

                for segment_id in tqdm(segment_ids_for_split, desc=f"  {split_name}"):
                    try:
                        # AUDIO FEATURES - Real COVAREP features
                        if audio_feature in hf and segment_id in hf[audio_feature]:
                            audio_data = hf[audio_feature][segment_id]['features'][()]
                            audio_data = np.nan_to_num(audio_data, nan=0.0)
                            audio_data = np.clip(audio_data, -10, 10)
                            audio_data = pad_or_truncate(audio_data, self.max_seq_len)
                        else:
                            continue
                        
                        # VISUAL FEATURES - Real FACET features
                        if visual_feature in hf and segment_id in hf[visual_feature]:
                            visual_data = hf[visual_feature][segment_id]['features'][()]
                            visual_data = np.nan_to_num(visual_data, nan=0.0)
                            visual_data = np.clip(visual_data, -10, 10)
                            visual_data = pad_or_truncate(visual_data, self.max_seq_len)
                        else:
                            continue
                        
                        # TEXT FEATURES - BERT embeddings
                        if text_feature in hf and segment_id in hf[text_feature]:
                            words_data = hf[text_feature][segment_id]['features'][()]
                            words = [w[0].decode('utf-8') if len(w) > 0 else '' for w in words_data]
                            words = [w for w in words if w and w != 'sp']
                            
                            # Create text features for each timestep
                            text_data = []
                            for t in range(self.max_seq_len):
                                # Use a sliding window of words
                                start_idx = max(0, t-2)
                                end_idx = min(len(words), t+3)
                                window_words = words[start_idx:end_idx]
                                
                                if window_words and self.bert_model is not None:
                                    # Use BERT embeddings
                                    text_features = self.get_bert_embeddings(window_words)
                                else:
                                    # Fallback to simple features
                                    text_features = self.create_simple_text_features(window_words)
                                
                                text_data.append(text_features)
                            
                            text_data = np.array(text_data, dtype=np.float32)
                        else:
                            continue
                        
                        # LABELS
                        if segment_id in hf[label_feature]:
                            label_value = hf[label_feature][segment_id]['features'][()][0]
                            
                            if label_value <= -1:
                                label = 0  # Negative
                            elif label_value >= 1:
                                label = 2  # Positive
                            else:
                                label = 1  # Neutral
                        else:
                            continue
                        
                        # Validate data
                        if (np.any(np.isnan(text_data)) or np.any(np.isnan(visual_data)) or 
                            np.any(np.isnan(audio_data))):
                            logger.warning(f"NaN detected in {segment_id}, skipping")
                            continue
                        
                        # Create sample
                        sample = {
                            'text': text_data,
                            'visual': visual_data,
                            'audio': audio_data,
                            'label': int(label)
                        }
                        
                        processed_data[split_name].append(sample)
                        
                    except Exception as e:
                        logger.error(f"Error processing segment {segment_id}: {str(e)}")
                        continue

        return processed_data

    def save_processed_data(self, processed_data):
        """Save processed data to pickle files."""
        processed_dir = self.data_dir / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving processed data...")
        for split_name, data in processed_data.items():
            output_path = processed_dir / f'{split_name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(data)} samples to {output_path}")

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
        logger.info("Starting BERT-Enhanced Data Preprocessing")
        logger.info("="*50)
        
        processed_data = self.load_from_hdf5()
        self.save_processed_data(processed_data)
        self.compute_statistics(processed_data)
        
        logger.info("\n✓ BERT preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(config_path='configs/config.yaml')
    preprocessor.run()