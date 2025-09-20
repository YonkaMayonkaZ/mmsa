import numpy as np
import mmdatasdk

def pad_or_truncate(features, max_len):
    """
    Pads or truncates a sequence of features to a specified length.

    Args:
        features (np.array): The feature array of shape (seq_len, feature_dim).
        max_len (int): The desired fixed sequence length.

    Returns:
        np.array: The padded or truncated feature array.
    """
    seq_len, feat_dim = features.shape
    if seq_len >= max_len:
        return features[:max_len]
    else:
        padding = np.zeros((max_len - seq_len, feat_dim))
        return np.vstack((features, padding))

def get_data_splits():
    """
    Fetches the standard train, validation, and test splits for CMU-MOSI
    using the CMU Multimodal SDK.

    Returns:
        dict: A dictionary containing lists of video IDs for 'train', 'valid', and 'test' splits.
    """
    sdk_standard_splits = mmdatasdk.cmu_mosi.standard_folds
    
    # The SDK provides IDs in a specific format, we just need the video IDs
    train_ids = [vid for vid in sdk_standard_splits['train']]
    valid_ids = [vid for vid in sdk_standard_splits['valid']]
    test_ids = [vid for vid in sdk_standard_splits['test']]
    
    return {
        'train': train_ids,
        'valid': valid_ids,
        'test': test_ids
    }
