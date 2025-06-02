import os
import sys

def list_pickles(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Error: “{root_dir}” is not a directory or does not exist.")
        sys.exit(1)

    print(f">>> Scanning for .pkl under: {root_dir}\n")
    found_any = False
    for dirpath, _, filenames in os.walk(root_dir):
        # This prints every directory as we visit it
        print(f"Visiting directory: {dirpath}")
        for fname in filenames:
            if fname.lower().endswith(".pkl"):
                found_any = True
                full_path = os.path.join(dirpath, fname)
                print(f"  → {full_path}")

    if not found_any:
        print("\nNo .pkl files found under that path.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_all_pkls.py /path/to/top_level_folder")
        sys.exit(1)

    list_pickles(sys.argv[1])