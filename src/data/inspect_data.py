#!/usr/bin/env python
"""
Inspect the CMU-MOSI HDF5 file to understand the actual data structure
"""
import h5py
import numpy as np

def inspect_hdf5(file_path='data/raw/mosi.hdf5'):
    """Thoroughly inspect the HDF5 file structure and content"""
    
    print("🔍 Inspecting CMU-MOSI HDF5 File Structure")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as hf:
        print(f"📁 Top-level groups: {list(hf.keys())}")
        print()
        
        # Get a sample segment ID for detailed inspection
        labels_group = hf['Opinion Segment Labels']
        sample_segment = list(labels_group.keys())[0]
        print(f"🔬 Inspecting sample segment: '{sample_segment}'")
        print()
        
        # Inspect each modality
        for group_name in hf.keys():
            print(f"📊 Group: {group_name}")
            group = hf[group_name]
            
            if sample_segment in group:
                seg_data = group[sample_segment]
                print(f"  ├── Available keys: {list(seg_data.keys())}")
                
                if 'features' in seg_data:
                    features = seg_data['features']
                    print(f"  ├── Features shape: {features.shape}")
                    print(f"  ├── Features dtype: {features.dtype}")
                    
                    # Show sample data for different modalities
                    if group_name == 'words':
                        # Text might be strings or word IDs
                        sample_data = features[()]
                        if len(sample_data) > 0:
                            print(f"  ├── Sample text data: {sample_data[:5]}...")
                    else:
                        # Numerical features
                        sample_data = features[()]
                        if len(sample_data.shape) > 1:
                            print(f"  ├── Sample features (first 3 timesteps, first 5 dims):")
                            print(f"  │   {sample_data[:3, :5]}")
                        else:
                            print(f"  ├── Sample features: {sample_data[:10]}")
                
                if 'intervals' in seg_data:
                    intervals = seg_data['intervals'][()]
                    print(f"  ├── Time intervals: {intervals}")
                    
            else:
                print(f"  ├── Sample segment not found in this group")
            
            print(f"  └── Total segments in group: {len(list(group.keys()))}")
            print()
        
        # Check if there are any other text feature groups
        print("🔍 Looking for text/language features...")
        text_related = [k for k in hf.keys() if any(word in k.lower() for word in ['text', 'word', 'lang', 'bert', 'glove'])]
        print(f"Text-related groups: {text_related}")
        print()
        
        # Summary statistics
        print("📈 Dataset Summary:")
        total_segments = len(list(labels_group.keys()))
        print(f"  ├── Total segments: {total_segments}")
        
        # Check feature dimensions for each modality
        for modality in ['COVAREP', 'FACET_4.2', 'words']:
            if modality in hf and sample_segment in hf[modality]:
                shape = hf[modality][sample_segment]['features'].shape
                print(f"  ├── {modality} dimensions: {shape}")
        
        print("=" * 60)

if __name__ == "__main__":
    inspect_hdf5()