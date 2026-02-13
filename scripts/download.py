import os
import subprocess
import sys

# --- CONFIGURATION ---
# 1. Determine directories
SCRIPT_PATH = os.path.abspath(__file__)          # D:/.../scripts/download_kits19.py
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)        # D:/.../scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)       # D:/.../KidneyCancerAI

# 2. Define the target list file
LIST_FILE = os.path.join(PROJECT_ROOT, "kits19_targets.txt")

def main():
    print(f"--- 🚀 Starting Turbo Downloader (Fixed) ---")
    
    # 3. CRITICAL FIX: Change working directory to Project Root
    # This ensures 'data/raw/...' is relative to the root, not the scripts folder.
    os.chdir(PROJECT_ROOT)
    print(f"📂 Working Directory set to: {os.getcwd()}")
    
    # 4. Generate the Aria2c Input File using RELATIVE PATHS
    print("📝 Generating download list...")
    
    try:
        with open(LIST_FILE, "w") as f:
            for i in range(210):
                # Format 0 to 00000
                case_id = f"{i:05d}"
                
                # Relative path from Project Root
                # e.g., data/raw/kits19/case_00000/imaging.nii.gz
                rel_dir = os.path.join("data", "raw", "kits19", f"case_{case_id}")
                rel_path = os.path.join(rel_dir, "imaging.nii.gz")
                
                # Create the folder first (Python handles this safely)
                os.makedirs(rel_dir, exist_ok=True)
                
                # Define URL
                url = f"https://kits19.sfo2.digitaloceanspaces.com/interpolated_{case_id}.nii.gz"
                
                # Write to aria2 input file
                f.write(f"{url}\n")
                f.write(f"  out={rel_path}\n")
                
        print(f"✅ Target list created: {LIST_FILE}")
        
    except Exception as e:
        print(f"❌ Error creating list file: {e}")
        return

    # 5. Call Aria2c
    print("\n🔥 Launching Aria2c...")
    
    cmd = [
        "aria2c",
        "-i", "kits19_targets.txt", # Input file (now in root)
        "-j", "8",                  # 8 downloads at once
        "-x", "16",                 # 16 connections per file
        "-s", "16",                 # Split file into 16 parts
        "-k", "1M",                 # Min split size
        "--console-log-level=warn",
        "--auto-file-renaming=false", # Prevent duplicates like imaging.1.nii.gz
        "--allow-overwrite=true"      # Overwrite if exists (useful for retries)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Download Complete!")
        
        # Cleanup
        if os.path.exists(LIST_FILE):
            os.remove(LIST_FILE)
            print("🧹 Cleanup: Removed target list.")
            
    except FileNotFoundError:
        print("\n❌ Error: 'aria2c' not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Aria2c error (Code {e.returncode}).")
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

if __name__ == "__main__":
    main()