import os
import glob
from PIL import Image

def diagnose_images(folder_path):
    print(f"--- DIAGNOSING FOLDER: {folder_path} ---")
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print("CRITICAL ERROR: Folder does not exist.")
        return

    files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if len(files) == 0:
        print("CRITICAL ERROR: No .png files found in this folder.")
        return

    print(f"Found {len(files)} files. Checking the first 5...")

    brokenFiles = []
    
    for i, fpath in enumerate(files[:800]):
        print(f"\nFile: {os.path.basename(fpath)}")
        
        # 1. Check File Size
        size_bytes = os.path.getsize(fpath)
        size_kb = size_bytes / 1024
        print(f"Size: {size_kb:.2f} KB")
        
        if size_bytes == 0:
            print("-> FAIL: File is empty (0 bytes). Re-download dataset.")
            continue
            
        # 2. Check Headers
        try:
            with open(fpath, 'rb') as f:
                header = f.read(8)
                # PNG signature: \x89PNG\r\n\x1a\n
                if header != b'\x89PNG\r\n\x1a\n':
                    print(f"-> FAIL: Invalid PNG Header. First 8 bytes: {header}")
                    brokenFiles.append({os.path.basename(fpath)})
                    continue
                else:
                    print("-> Header: OK (Valid PNG signature)")
        except Exception as e:
            print(f"-> FAIL: Could not read file bytes. Error: {e}")
            brokenFiles.append({os.path.basename(fpath)})
            continue

        # 3. Try PIL Load
        try:
            with Image.open(fpath) as img:
                img.load() # Force load pixel data
                print(f"-> PIL Load: OK ({img.size}, {img.mode})")
        except Exception as e:
            print(f"-> PIL FAIL: {e}")
            brokenFiles.append({os.path.basename(fpath)})



    print(brokenFiles)


# Run on your Train HR path
train_hr = "/home/bo22354/Year 4/AI/AVAI/data/Train/DIV2K_train_HR"
diagnose_images(train_hr)