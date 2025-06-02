import os
import sys
import pickle
import numpy as np

def count_detections_from_main(pkl_obj, file_path):
    """
    Given a loaded pickle dict with key 'main', convert to numpy array,
    drop columns 0–3, filter rows where agentness > 0.5, and return row count.
    """
    if not isinstance(pkl_obj, dict):
        raise ValueError(f"Expected a dict in \"{file_path}\", got {type(pkl_obj)}.")

    if "main" not in pkl_obj:
        raise ValueError(f"No key 'main' in dict from \"{file_path}\".")

    arr = pkl_obj["main"]
    arr = np.asarray(arr)               # ensure numpy
    if arr.ndim < 2 or arr.shape[1] < 5:
        raise ValueError(
            f"Array under 'main' must have at least 5 columns in \"{file_path}\"."
        )

    arr = arr[:, 4:]                    # drop the first 4 columns (bbox coords)
    arr = arr[arr[:, 0] > 0.5]          # only keep rows where agentness > 0.5
    return arr.shape[0]                 # number of remaining rows

def main(root_dir):
    #— DEBUG: print the root_dir you actually received
    print(f">>> Starting walk at root_dir = {root_dir}\n")

    if not os.path.isdir(root_dir):
        print(f"Error: \"{root_dir}\" is not a directory or does not exist.")
        sys.exit(1)

    all_counts = []
    per_file_counts = []

    # os.walk will recurse into EVERY subfolder under root_dir
    for dirpath, _, filenames in os.walk(root_dir):
        #— DEBUG: show each directory as we enter it
        print(f"Visiting directory: {dirpath}")

        for fname in filenames:
            if not fname.lower().endswith(".pkl"):
                continue

            full_path = os.path.join(dirpath, fname)
            try:
                with open(full_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load \"{full_path}\": {e}")
                continue

            try:
                num_dets = count_detections_from_main(data, full_path)
            except ValueError as e:
                print(f"[WARNING] Skipping \"{full_path}\": {e}")
                continue

            per_file_counts.append((full_path, num_dets))
            all_counts.append(num_dets)

    if not per_file_counts:
        print("\nNo valid .pkl files with a usable 'main' array found.")
        return

    # Print per‐file results
    print("\nPer‐file detection counts (filtered by agentness > 0.5):")
    for path, cnt in per_file_counts:
        print(f"  {path}: {cnt}")

    # Compute overall stats
    minimum = min(all_counts)
    maximum = max(all_counts)
    average = sum(all_counts) / len(all_counts)

    print("\nOverall statistics across all .pkl files:")
    print(f"  Minimum detections: {minimum}")
    print(f"  Maximum detections: {maximum}")
    print(f"  Average detections: {average:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_detections_main_debug.py /path/to/root_folder")
        sys.exit(1)

    root_directory = sys.argv[1]
    main(root_directory)