"""
Data utilities for CMU-MOSI preprocessing - Dynamic split version
"""
import numpy as np

def pad_or_truncate(features, max_len):
    """
    Pads or truncates a sequence of features to a specified length.
    """
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    seq_len, feat_dim = features.shape
    if seq_len >= max_len:
        return features[:max_len]
    else:
        padding = np.zeros((max_len - seq_len, feat_dim))
        return np.vstack((features, padding))

def get_data_splits_from_segments(all_segment_ids):
    """
    Create train/valid/test splits dynamically from actual segment IDs in the dataset.
    
    Args:
        all_segment_ids (list): All segment IDs found in the HDF5 file
        
    Returns:
        dict: Split assignments for each segment ID
    """
    # Extract unique video IDs from segment IDs
    video_ids = list(set([sid.split('[')[0] for sid in all_segment_ids]))
    video_ids.sort()  # For reproducibility
    
    print(f"Found {len(video_ids)} unique videos: {video_ids[:10]}...")  # Show first 10
    
    # Split videos: 70% train, 15% valid, 15% test
    n_videos = len(video_ids)
    n_train = int(0.7 * n_videos)
    n_valid = int(0.15 * n_videos)
    
    train_videos = video_ids[:n_train]
    valid_videos = video_ids[n_train:n_train + n_valid]
    test_videos = video_ids[n_train + n_valid:]
    
    print(f"Split: {len(train_videos)} train, {len(valid_videos)} valid, {len(test_videos)} test videos")
    
    return {
        'train': train_videos,
        'valid': valid_videos,
        'test': test_videos
    }

def get_data_splits():
    """
    Fallback function that returns the standard splits.
    This is kept for compatibility but should use get_data_splits_from_segments() instead.
    """
    train_ids = [
        '2iD-tVS8NPw', '8d-gEyoeBzc', 'Qg1p4z1hpvo', 'BioHAh1qJAQ', 
        'GWuJjcEuzt8', 'I5FSYVGhJfA', 'J5MWW6HgH4Y', 'LSi-o-IrDJs',
        'OQvJTdtJ2H4', 'PZ-lDQFJhS8', 'TvyZBvOMOTc', 'VbQk4H8hgr0',
        '_dI--eQ6qVU', 'c5xsKMxpXnc', 'd6hH302o4v8', 'f_pcplsH4Jw',
        'iiK8YX8oH1E', 'kLa2zQpPgp8', 'lvjOW8iGhHI', 'lviIkYiafWo',
        'mQ055hHdxbE', 'nbWiPyCm4g0', 'phBUpBr1hSo', 'tmZoasNr4rU',
        'vSRW_L_Ah14', 'wMbj5Kp8T_A', 'x0q0el4_NrI'
    ]
    
    valid_ids = [
        'HEsqSxL_c2s', 'OtVxDvYrj8Y', 'VbcXkxXjKJ0', '_dVFknCdNYg',
        'aiEXnCPZubE', 'c7UH_rxdZv4', 'f9O3YtCgyrw', 'jHpVoFf_12Q',
        'nbKMnIWP23M', 'nzpVDcQ0ywI', 'oUzmvn3FQCU', 'v4OjMhR5-tQ'
    ]
    
    test_ids = [
        'BC9XOoKn5DI', 'h3UEqsqG-ec', 'AtUQYDFr52E', 'VVBhyz5nG8w',
        '_yCtJzU9CYw', 'c9xIDGe7_ls', 'f9owS5LkQa8', 'jK9lVHGXPfE',
        'nj3uV1hMYVY', 'o6RWUfcxrUg', 'nXXwU_o_VdE', 'vAhrbDX5SLw'
    ]
    
    return {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }