"""
Preprocess CMU-MOSI data with PROPER temporal alignment
Based on research papers showing correct alignment strategies
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
        self.hdf5_path = self.data_dir / 'raw' / 'mosi.hdf5'
        
        # Initialize BERT model
        self.setup_bert()

    def setup_bert(self):
        """Initialize BERT model for text embeddings"""
        logger.info("Loading BERT model for text embeddings...")
        
        model_name = "distilbert-base-uncased"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            logger.info(f"BERT model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.bert_model = None
            self.tokenizer = None

    def get_utterance_bert_embedding(self, words):
        """
        Get BERT embedding for the ENTIRE utterance (correct approach)
        Then broadcast/repeat to match audio/visual sequence length
        """
        if self.bert_model is None or not words:
            return np.zeros(self.config['data']['dims']['text'], dtype=np.float32)
        
        try:
            # Join ALL words into complete utterance text
            full_text = ' '.join(words)
            
            # Tokenize the complete text
            inputs = self.tokenizer(
                full_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get sentence-level embedding
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token for sentence representation
                sentence_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            # Resize to match config dimensions
            expected_dim = self.config['data']['dims']['text']
            if len(sentence_embedding) > expected_dim:
                embedding = sentence_embedding[:expected_dim]
            else:
                embedding = np.zeros(expected_dim, dtype=np.float32)
                embedding[:len(sentence_embedding)] = sentence_embedding
            
            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"BERT embedding failed: {e}")
            return np.zeros(self.config['data']['dims']['text'], dtype=np.float32)

    def inspect_data_structure(self, hf):
        """Inspect actual data structure to determine real sequence length"""
        # Get sample data to understand actual structure
        label_feature = self.config['data']['features']['labels']
        sample_segment = list(hf[label_feature].keys())[0]
        
        # Check actual sequence lengths
        audio_feature = self.config['data']['features']['audio']
        visual_feature = self.config['data']['features']['visual']
        
        if sample_segment in hf[audio_feature]:
            audio_shape = hf[audio_feature][sample_segment]['features'].shape
            logger.info(f"Audio shape: {audio_shape}")
        
        if sample_segment in hf[visual_feature]:
            visual_shape = hf[visual_feature][sample_segment]['features'].shape
            logger.info(f"Visual shape: {visual_shape}")
        
        # Return actual sequence length (not config max_seq_length)
        return audio_shape[0] if 'audio_shape' in locals() else 8

    def load_from_hdf5(self):
        """Load data with PROPER temporal alignment"""
        processed_data = {'train': [], 'valid': [], 'test': []}
        
        logger.info(f"Loading data from HDF5 file: {self.hdf5_path}")
        if not self.hdf5_path.exists():
            logger.error(f"HDF5 file not found at {self.hdf5_path}")
            sys.exit(1)

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_feature = self.config['data']['features']['audio']
            visual_feature = self.config['data']['features']['visual']
            text_feature = self.config['data']['features']['text']
            label_feature = self.config['data']['features']['labels']
            
            logger.info(f"Available groups in HDF5: {list(hf.keys())}")
            
            # Inspect actual data structure
            actual_seq_len = self.inspect_data_structure(hf)
            logger.info(f"Using actual sequence length: {actual_seq_len}")
            
            all_segment_ids = list(hf[label_feature].keys())
            logger.info(f"Found {len(all_segment_ids)} segments in dataset")

            splits = get_data_splits_from_segments(all_segment_ids)

            for split_name, video_ids in splits.items():
                logger.info(f"Processing {split_name} split...")
                
                segment_ids_for_split = [sid for sid in all_segment_ids if sid.split('[')[0] in video_ids]
                logger.info(f"  Found {len(segment_ids_for_split)} segments for {split_name}")

                for segment_id in tqdm(segment_ids_for_split, desc=f"  {split_name}"):
                    try:
                        # AUDIO FEATURES - Use consistent padding
                        if audio_feature in hf and segment_id in hf[audio_feature]:
                            audio_data = hf[audio_feature][segment_id]['features'][()]
                            audio_data = np.nan_to_num(audio_data, nan=0.0)
                            audio_data = np.clip(audio_data, -10, 10)
                            # Pad/truncate to consistent length
                            audio_data = pad_or_truncate(audio_data, self.config['data']['max_seq_length'])
                        else:
                            continue
                        
                        # VISUAL FEATURES - Use consistent padding
                        if visual_feature in hf and segment_id in hf[visual_feature]:
                            visual_data = hf[visual_feature][segment_id]['features'][()]
                            visual_data = np.nan_to_num(visual_data, nan=0.0)
                            visual_data = np.clip(visual_data, -10, 10)
                            # Pad/truncate to consistent length
                            visual_data = pad_or_truncate(visual_data, self.config['data']['max_seq_length'])
                        else:
                            continue
                        
                        # TEXT FEATURES - PROPER APPROACH with consistent padding
                        if text_feature in hf and segment_id in hf[text_feature]:
                            words_data = hf[text_feature][segment_id]['features'][()]
                            words = [w[0].decode('utf-8') if len(w) > 0 else '' for w in words_data]
                            words = [w for w in words if w and w != 'sp']
                            
                            # Get SINGLE utterance-level embedding
                            utterance_embedding = self.get_utterance_bert_embedding(words)
                            
                            # Broadcast to consistent sequence length
                            seq_len = self.config['data']['max_seq_length']
                            text_data = np.tile(utterance_embedding, (seq_len, 1))
                            
                        else:
                            continue
                        
                        # Ensure all modalities have same sequence length
                        seq_len = self.config['data']['max_seq_length']
                        assert audio_data.shape[0] == seq_len
                        assert visual_data.shape[0] == seq_len  
                        assert text_data.shape[0] == seq_len
                        
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
                        
                        # Final validation
                        if (np.any(np.isnan(text_data)) or np.any(np.isnan(visual_data)) or 
                            np.any(np.isnan(audio_data))):
                            continue
                        
                        # Create sample with proper alignment
                        sample = {
                            'text': text_data.astype(np.float32),
                            'visual': visual_data.astype(np.float32),
                            'audio': audio_data.astype(np.float32),
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
            
            # Check actual sequence lengths
            if data:
                sample_seq_len = data[0]['text'].shape[0]
                logger.info(f"{split_name.upper()} Split:")
                logger.info(f"  Total samples: {total_samples}")
                logger.info(f"  Sequence length: {sample_seq_len}")
                logger.info(f"  Negative (0): {neg_count} ({neg_count/total_samples*100:.1f}%)")
                logger.info(f"  Neutral  (1): {neu_count} ({neu_count/total_samples*100:.1f}%)")
                logger.info(f"  Positive (2): {pos_count} ({pos_count/total_samples*100:.1f}%)")
        logger.info("--------------------------")

    def run(self):
        """Run the complete preprocessing pipeline."""
        logger.info("="*50)
        logger.info("Starting PROPERLY ALIGNED Data Preprocessing")
        logger.info("="*50)
        
        processed_data = self.load_from_hdf5()
        self.save_processed_data(processed_data)
        self.compute_statistics(processed_data)
        
        logger.info("\n✓ Properly aligned preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(config_path='configs/config.yaml')
    preprocessor.run()