"""
Preprocess CMU-MOSI data with PROPER temporal alignment.
This version saves pre-computed, word-level BERT embeddings that are
aligned with the audio and visual modalities.
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

        # Initialize BERT for embedding extraction
        self.setup_bert()

    def setup_bert(self):
        """Initialize BERT model and tokenizer for creating embeddings."""
        logger.info("Loading BERT model for word-level embeddings...")
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.bert_model.eval()
        logger.info(f"BERT model loaded on {self.device}")

    def get_word_level_embeddings(self, words):
        """
        Generates word-level embeddings and aligns them with the original words.
        Handles sub-word tokens by averaging them.
        """
        if not words:
            return np.zeros((0, self.config['data']['dims']['text']))

        # Tokenize and get embeddings from BERT
        inputs = self.tokenizer(words, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)
        
        # CORRECT WAY to move tensors to device, preserving the object type
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            token_embeddings = outputs.last_hidden_state[0]

        # Map token embeddings back to word embeddings
        word_embeddings = []
        word_ids = inputs.word_ids(batch_index=0)
        
        previous_word_idx = None
        current_word_tokens = []
        
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:  # Special token ([CLS], [SEP])
                continue
            
            if word_idx != previous_word_idx:
                if current_word_tokens:
                    word_embeddings.append(torch.mean(torch.stack(current_word_tokens), dim=0))
                current_word_tokens = []
            
            current_word_tokens.append(token_embeddings[token_idx])
            previous_word_idx = word_idx
        
        if current_word_tokens:
            word_embeddings.append(torch.mean(torch.stack(current_word_tokens), dim=0))
            
        return torch.stack(word_embeddings).cpu().numpy()


    def load_from_hdf5(self):
        """Load data and save aligned, pre-computed word embeddings."""
        processed_data = {'train': [], 'valid': [], 'test': []}
        logger.info(f"Loading data from HDF5 file: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_feature = self.config['data']['features']['audio']
            visual_feature = self.config['data']['features']['visual']
            text_feature = self.config['data']['features']['text']
            label_feature = self.config['data']['features']['labels']
            all_segment_ids = list(hf[label_feature].keys())
            splits = get_data_splits_from_segments(all_segment_ids)

            for split_name, video_ids in splits.items():
                logger.info(f"Processing {split_name} split...")
                segment_ids_for_split = [sid for sid in all_segment_ids if sid.split('[')[0] in video_ids]

                for segment_id in tqdm(segment_ids_for_split, desc=f"  {split_name}"):
                    try:
                        # Get raw words
                        if text_feature in hf and segment_id in hf[text_feature]:
                            words_data = hf[text_feature][segment_id]['features'][()]
                            words = [w[0].decode('utf-8') for w in words_data if len(w) > 0 and w[0].decode('utf-8') != 'sp']
                        else: continue

                        # TEXT: Generate and pad word-level embeddings
                        text_data = self.get_word_level_embeddings(words)
                        text_data = pad_or_truncate(text_data, self.config['data']['max_seq_length'])
                        
                        # AUDIO: Pad to the same length
                        if audio_feature in hf and segment_id in hf[audio_feature]:
                            audio_data = hf[audio_feature][segment_id]['features'][()]
                            audio_data = np.nan_to_num(audio_data, nan=0.0)
                            audio_data = pad_or_truncate(audio_data, self.config['data']['max_seq_length'])
                        else: continue

                        # VISUAL: Pad to the same length
                        if visual_feature in hf and segment_id in hf[visual_feature]:
                            visual_data = hf[visual_feature][segment_id]['features'][()]
                            visual_data = np.nan_to_num(visual_data, nan=0.0)
                            visual_data = pad_or_truncate(visual_data, self.config['data']['max_seq_length'])
                        else: continue

                        # LABELS
                        if segment_id in hf[label_feature]:
                            label_value = hf[label_feature][segment_id]['features'][()][0]
                            label = 1 # Neutral as default
                            if label_value <= -1: label = 0  # Negative
                            elif label_value >= 1: label = 2  # Positive
                        else: continue

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

    def run(self):
        """Run the complete preprocessing pipeline."""
        logger.info("="*50)
        logger.info("Starting Data Preprocessing (Aligned Word-Level Embeddings)")
        logger.info("="*50)
        processed_data = self.load_from_hdf5()
        self.save_processed_data(processed_data)
        logger.info("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(config_path='configs/config.yaml')
    preprocessor.run()


