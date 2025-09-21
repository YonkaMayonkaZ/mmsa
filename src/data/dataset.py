import h5py
import numpy as np

# Make sure 'mosi.hdf5' is in the same directory as your notebook,
# or provide the full path to the file.
file_path = 'path/to/your/mosi.hdf5'

try:
    with h5py.File(file_path, 'r') as hf:
        print(f"Successfully opened: {file_path}")
        print("-" * 30)

        # --- Step 1: List top-level groups (modalities) ---
        print("Top-level keys (groups) in the file:")
        top_level_keys = list(hf.keys())
        print(top_level_keys)
        print("-" * 30)

        # --- Step 2: Explore the structure recursively ---
        print("Exploring the full data structure:")
        def print_structure(name, obj):
            # Indent based on the depth of the group/dataset
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent} L Dataset: {name.split('/')[-1]} | Shape: {obj.shape} | Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent} > Group: {name.split('/')[-1] or 'ROOT'}")

        hf.visititems(print_structure)
        print("-" * 30)

        # --- Step 3: Inspect a specific video segment ---
        # Let's assume the structure is something like 'text/video_id/features'
        # We'll try to access data for a sample video ID.
        # Replace '2iD-tVS8NPw_1' with an actual segment ID from your HDF5 file structure.
        if top_level_keys:
            first_group_name = top_level_keys[0]
            if isinstance(hf[first_group_name], h5py.Group) and len(hf[first_group_name].keys()) > 0:
                sample_segment_id = list(hf[first_group_name].keys())[0]
                print(f"Inspecting a sample segment: '{sample_segment_id}'")

                for group_name in top_level_keys:
                    try:
                        # Access the 'features' dataset for this segment
                        features = hf[f'{group_name}/{sample_segment_id}/features'][()]
                        print(f"  > Modality: {group_name}")
                        print(f"    - Features Shape: {features.shape}")
                        # print(f"    - Sample Features: \n{features[:2]}") # Uncomment to see first 2 feature vectors
                    except KeyError:
                        print(f"  > Could not find features for segment '{sample_segment_id}' in group '{group_name}'")
                    except Exception as e:
                        print(f"An error occurred while accessing {group_name}: {e}")

except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")