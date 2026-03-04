"""
Dataset Image Extractor for CDSS
---------------------------------
Run this script after cloning to extract the dataset images from the zip archive.
    python setup_dataset.py
"""
import zipfile
import os
import sys

ARCHIVE = os.path.join("Dataset", "images.zip")
DEST = os.path.join("Dataset", "images")

if not os.path.exists(ARCHIVE):
    print(f"ERROR: {ARCHIVE} not found. Make sure Git LFS pulled the file.")
    print("Run: git lfs pull")
    sys.exit(1)

if os.path.exists(DEST) and len(os.listdir(DEST)) > 0:
    print(f"Dataset/images/ already contains {len(os.listdir(DEST))} files. Skipping extraction.")
    sys.exit(0)

os.makedirs(DEST, exist_ok=True)
print(f"Extracting {ARCHIVE} → {DEST}/ ...")

with zipfile.ZipFile(ARCHIVE, 'r') as zf:
    total = len(zf.namelist())
    for i, member in enumerate(zf.namelist(), 1):
        zf.extract(member, DEST)
        if i % 2000 == 0 or i == total:
            print(f"  {i}/{total} files extracted ({100*i//total}%)")

print(f"Done! {total} images extracted to {DEST}/")
