import os
import tarfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

parent_folder = Path('/scratch/gpfs/JHA/mb5157/large_data/diabetes_2025_6years')
destination_dir = parent_folder / "paper_archives"
unpack_dir      = parent_folder / "papers"
bad_archives    = parent_folder / "archives_with_error"
max_workers     = 6  # tune to number of CPU cores

# Ensure directories exist
unpack_dir.mkdir(parents=True, exist_ok=True)
bad_archives.mkdir(parents=True, exist_ok=True)

def extract_archive(tar_path):
    """
    Attempts to extract tar_path into unpack_dir.
    Returns (tar_path, success_flag, error_message).
    """
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=unpack_dir)
        return (tar_path, True, "")
    except Exception as e:
        return (tar_path, False, str(e))

def main():
    tar_files = list(destination_dir.glob("*.tar*"))
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_archive, tf): tf for tf in tar_files}
        for future in as_completed(futures):
            tar_path, success, error = future.result()
            if success:
                print(f"Extracted: {tar_path.name}")
            else:
                print(f"Failed: {tar_path.name}, moving to bad_archives")
                tar_path.replace(bad_archives / tar_path.name)

if __name__ == "__main__":
    main()

