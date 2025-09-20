#!/usr/bin/env python
"""
Download CMU-MOSI dataset using MMSDK
"""
import os
import sys
import json
import pickle
from pathlib import Path
import mmdatasdk
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CMU-MOSI feature URLs
MOSI_FEATURES = {
    'text': 'http://multicomp.cs.cmu.edu/CMU-MOSI/data/CMU_MOSI_TimestampedWordVectors.csd',
    'visual': 'http://multicomp.cs.cmu.edu/CMU-MOSI/data/CMU_MOSI_VisualFacet_4.1.csd',
    'audio': 'http://multicomp.cs.cmu.edu/CMU-MOSI/data/CMU_MOSI_COVAREP.csd',
    'labels': 'http://multicomp.cs.cmu.edu/CMU-MOSI/data/CMU_MOSI_Opinion_Labels.csd'
}

def download_mosi_data(output_dir='/workspace/data/raw'):
    """Download CMU-MOSI dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*50)
    logger.info("CMU-MOSI Dataset Download")
    logger.info("="*50)
    
    # Check if already downloaded
    cache_file = output_dir / 'mosi_raw.pkl'
    if cache_file.exists():
        logger.info("Dataset already downloaded. Loading from cache...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        logger.info("Loaded from cache successfully!")
        return dataset
    
    # Download dataset
    logger.info("Downloading CMU-MOSI features...")
    logger.info("This will download approximately 500MB of data")
    
    start_time = datetime.now()
    
    try:
        dataset = mmdatasdk.mmdataset(MOSI_FEATURES)
        logger.info(f"✓ Download complete in {datetime.now() - start_time}")
        
        # Save raw dataset
        logger.info("Saving raw dataset to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save metadata
        metadata = {
            'download_time': str(datetime.now()),
            'features': list(MOSI_FEATURES.keys()),
            'num_segments': len(dataset.keys()),
            'feature_dims': {
                'text': 300,
                'visual': 35,
                'audio': 74
            }
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Dataset saved to {cache_file}")
        logger.info(f"✓ Number of segments: {len(dataset.keys())}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    dataset = download_mosi_data()
    print(f"\n✓ Download complete! Dataset contains {len(dataset.keys())} segments")