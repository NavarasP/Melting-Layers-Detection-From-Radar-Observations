#!/usr/bin/env python3
"""
Check for corrupted HDF5 files in the dataset and optionally remove them.
"""
import os
import glob
import h5py
import argparse

def check_file(filepath):
    """Test if an HDF5 file can be opened."""
    try:
        with h5py.File(filepath, 'r') as hf:
            # Try to access keys to verify integrity
            _ = list(hf.keys())
        return True, None
    except (OSError, IOError, RuntimeError) as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Check HDF5 files for corruption')
    parser.add_argument('--data-dir', default='GPM_DPR_2024', help='Data directory')
    parser.add_argument('--pattern', default='**/*.HDF5', help='File pattern')
    parser.add_argument('--remove', action='store_true', help='Remove corrupted files')
    args = parser.parse_args()
    
    # Find all files
    pattern_path = os.path.join(args.data_dir, args.pattern)
    files = glob.glob(pattern_path, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    
    print(f"Checking {len(files)} HDF5 files in {args.data_dir}...")
    print("-" * 70)
    
    corrupted = []
    valid = 0
    
    for filepath in files:
        is_valid, error = check_file(filepath)
        if is_valid:
            valid += 1
        else:
            corrupted.append((filepath, error))
            print(f"✗ CORRUPTED: {os.path.basename(filepath)}")
            print(f"  Error: {error}")
            if args.remove:
                try:
                    os.remove(filepath)
                    print(f"  → Removed")
                except Exception as e:
                    print(f"  → Failed to remove: {e}")
    
    print("-" * 70)
    print(f"\nResults:")
    print(f"  Valid files: {valid}")
    print(f"  Corrupted files: {len(corrupted)}")
    
    if corrupted and not args.remove:
        print(f"\nTo remove corrupted files, run:")
        print(f"  python check_corrupted_files.py --remove")

if __name__ == '__main__':
    main()
