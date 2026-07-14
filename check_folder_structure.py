
from pathlib import Path
import os
from collections import defaultdict

# FolderName = "E"
# #base_path = Path("data/Aquafin data (vertrouwelijk NDA)/microscopie historische foto's (en aanverwante documenten)")
# base_path= Path("data/Aquafin data (Sorted)/microscopie historische foto's (en aanverwante documenten)")
# src_root = base_path / FolderName

FolderName = "microscopie historische foto's (en aanverwante documenten)"
base_path = Path("data/Aquafin data (Sorted)")
#base_path= Path("data/Aquafin data (Sorted)/microscopie historische foto's (en aanverwante documenten)")
src_root = base_path / FolderName

def get_all_files(folder_path):
    """Recursively get all files in a folder with their relative paths"""
    files = {}
    if not folder_path.exists():
        print(f"Warning: {folder_path} does not exist")
        return files
    
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = Path(root) / filename
            rel_path = full_path.relative_to(folder_path) # Get path relative to the folder root
            files[str(rel_path)] = full_path # Store the full path for later use (e.g., to get file size)
    return files

def compare_folders():
    """Compare E folder with all parent folders"""
    
    # Get all files in E folder
    e_files = get_all_files(src_root)
    print(f"\n{'='*80}")
    print(f"E FOLDER ANALYSIS")
    print(f"{'='*80}")
    print(f"Total files in E: {len(e_files)}\n")
    
    # Get parent folders
    parent_path = src_root.parent
    parent_folders = []
    
    for item in sorted(parent_path.iterdir()): 
        if item.is_dir() and item.name != FolderName: 
            parent_folders.append(item) 
    
    print(f"Parent folders found: {len(parent_folders)}")
    for folder in parent_folders:
        print(f"  - {folder.name}")
    print()
    
    # Compare with each parent folder
    results = {
        "copied_files": defaultdict(list),  # files that exist in both E and parent
        "new_files": [],  # files only in E
        "file_stats": {}  # file details
    }
    
    for parent_folder in parent_folders:
        parent_files = get_all_files(parent_folder)
        folder_name = parent_folder.name
        results["file_stats"][folder_name] = len(parent_files)
        
        print(f"\n{'-'*80}")
        print(f"Comparing E with folder: {folder_name}")
        print(f"{'-'*80}")
        print(f"Total files in {folder_name}: {len(parent_files)}")
        
        # Find copied files (same path in both folders)
        copied = []
        for file_path in e_files.keys():
            if file_path in parent_files:
                copied.append(file_path)
                results["copied_files"][file_path].append(folder_name)
        print(f"Copied files from {folder_name} to E: {len(copied)}")
    
    # Identify files unique to E
    all_parent_files = set()
    for parent_folder in parent_folders:
        parent_files = get_all_files(parent_folder)
        all_parent_files.update(parent_files.keys())
    
    new_files = [f for f in e_files.keys() if f not in all_parent_files] 
    results["new_files"] = new_files
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total files in E: {len(e_files)}")
    print(f"Copied files (exist in parent folders): {len(set(results['copied_files'].keys()))}")
    print(f"New files (only in E): {len(new_files)}\n")
    
    if new_files:
        print(f"NEW FILES IN E (not found in any parent folder):")
        print(f"{'-'*80}")
        # for f in sorted(new_files):
        #     file_size = e_files[f].stat().st_size
        #     size_mb = file_size / (1024*1024)
        #     print(f"  📄 {f} ({size_mb:.2f} MB)")
    
    # Show which files are in multiple parent folders
    print(f"\n{'='*80}")
    print(f"FILES COPIED FROM MULTIPLE PARENT FOLDERS")
    print(f"{'='*80}")
    multi_parent = {f: parents for f, parents in results["copied_files"].items() if len(parents) > 1}
    if multi_parent:
        for f, parents in sorted(multi_parent.items()):
            print(f"  {f}")
            print(f"    Found in: {', '.join(sorted(parents))}")
    else:
        print("No files found in multiple parent folders")
    
    return results

if __name__ == "__main__":
    results = compare_folders()
