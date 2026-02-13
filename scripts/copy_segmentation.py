import os
import shutil
import sys

# --- CONFIGURATION ---
# 1. Determine paths
SCRIPT_PATH = os.path.abspath(__file__)           # scripts/copy_segmentations.py
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)         # scripts/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)        # KidneyCancerAI/

# Source: The cloned GitHub repo folder
SOURCE_BASE = os.path.join(PROJECT_ROOT, "kits19", "data")

# Destination: Your clean raw data folder
DEST_BASE = os.path.join(PROJECT_ROOT, "data", "raw", "kits19")

def main():
    print("--- 📦 Moving Segmentations to Raw Data Folder ---")
    print(f"📂 Source: {SOURCE_BASE}")
    print(f"📂 Dest:   {DEST_BASE}")
    print("-" * 50)

    # Check if source exists
    if not os.path.exists(SOURCE_BASE):
        print(f"❌ Error: Source directory not found: {SOURCE_BASE}")
        print("   Did you clone the repo? (git clone https://github.com/neheller/kits19)")
        return

    count = 0
    missing = 0

    # Loop through cases 0 to 209 (KiTS19 Training Set)
    for i in range(210):
        case_id = f"case_{i:05d}"
        
        src_file = os.path.join(SOURCE_BASE, case_id, "segmentation.nii.gz")
        dest_dir = os.path.join(DEST_BASE, case_id)
        dest_file = os.path.join(dest_dir, "segmentation.nii.gz")

        # 1. Check if the segmentation exists in the source
        if os.path.exists(src_file):
            
            # 2. Ensure destination folder exists (it should, from the download step)
            os.makedirs(dest_dir, exist_ok=True)

            # 3. Copy the file
            try:
                shutil.copy2(src_file, dest_file)
                print(f"✅ Copied: {case_id}")
                count += 1
            except Exception as e:
                print(f"❌ Failed to copy {case_id}: {e}")
        else:
            print(f"⚠️  Missing in source: {case_id}")
            missing += 1

    print("-" * 50)
    print(f"🎉 Success! Transferred {count} segmentation files.")
    if missing > 0:
        print(f"ℹ️  {missing} cases did not have segmentations (this is normal for test data or unreleased cases).")

if __name__ == "__main__":
    main()