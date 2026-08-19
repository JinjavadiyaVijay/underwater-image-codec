"""Script to organize raw EUVP and SUIM downloads into the structure expected by UWCodec."""

import os
import shutil
import random
from pathlib import Path

def get_image_files(directory):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in extensions:
                files.append(Path(root) / f)
    return files

def organize_euvp(datasets_root: Path):
    print("Organizing EUVP...")
    euvp_root = datasets_root / "EUVP"
    if not euvp_root.exists():
        print("  EUVP folder not found.")
        return

    train_dir = euvp_root / "train"
    val_dir = euvp_root / "val"
    test_dir = euvp_root / "test"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Gather trainA (distorted) images from Paired and Unpaired
    train_files = []
    val_files = []
    test_files = []

    for sub in ["Paired", "Unpaired"]:
        sub_dir = euvp_root / sub
        if not sub_dir.exists(): continue
        
        # In Paired, there are underwater_dark, underwater_imagenet, underwater_scenes
        # In Unpaired, there are trainA, trainB, validation directly
        if sub == "Paired":
            for category in ["underwater_dark", "underwater_imagenet", "underwater_scenes"]:
                cat_dir = sub_dir / category
                if not cat_dir.exists(): continue
                train_files.extend(get_image_files(cat_dir / "trainA"))
                val_files.extend(get_image_files(cat_dir / "validation"))
        else:
            train_files.extend(get_image_files(sub_dir / "trainA"))
            val_files.extend(get_image_files(sub_dir / "validation"))

    # test_samples
    test_samples_dir = euvp_root / "test_samples"
    if test_samples_dir.exists():
        test_files.extend(get_image_files(test_samples_dir))

    # Move files
    def move_files(files, target_dir, prefix):
        for i, f in enumerate(files):
            new_name = f"{prefix}_{i:05d}{f.suffix}"
            target_path = target_dir / new_name
            # If the file is already there, we might have run this before
            if not target_path.exists():
                shutil.move(str(f), str(target_path))

    move_files(train_files, train_dir, "train")
    print(f"  Moved {len(train_files)} images to EUVP/train")
    
    move_files(val_files, val_dir, "val")
    print(f"  Moved {len(val_files)} images to EUVP/val")
    
    move_files(test_files, test_dir, "test")
    print(f"  Moved {len(test_files)} images to EUVP/test")

    # Cleanup old dirs
    for d in ["Paired", "Unpaired", "test_samples", "eval_data"]:
        old_dir = euvp_root / d
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

def organize_suim(datasets_root: Path):
    print("Organizing SUIM...")
    suim_root = datasets_root / "SUIM"
    suim_root.mkdir(exist_ok=True)
    
    train_dir = suim_root / "train"
    val_dir = suim_root / "val"
    test_dir = suim_root / "test"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # train_val folder
    raw_train_val = datasets_root / "train_val" / "images"
    if raw_train_val.exists():
        all_train_files = get_image_files(raw_train_val)
        random.seed(42)
        random.shuffle(all_train_files)
        
        # Split 10% for validation
        val_count = int(len(all_train_files) * 0.1)
        val_files = all_train_files[:val_count]
        train_files = all_train_files[val_count:]
        
        for f in train_files:
            shutil.move(str(f), str(train_dir / f.name))
        print(f"  Moved {len(train_files)} images to SUIM/train")
        
        for f in val_files:
            shutil.move(str(f), str(val_dir / f.name))
        print(f"  Moved {len(val_files)} images to SUIM/val")

    # TEST folder
    raw_test = datasets_root / "TEST" / "images"
    if raw_test.exists():
        test_files = get_image_files(raw_test)
        for f in test_files:
            shutil.move(str(f), str(test_dir / f.name))
        print(f"  Moved {len(test_files)} images to SUIM/test")

    # Cleanup SUIM old dirs
    for d in ["train_val", "TEST", "Benchmark_Evaluation", "Checkpoint_Data"]:
        old_dir = datasets_root / d
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)

if __name__ == "__main__":
    datasets_root = Path("s:/IMG_compressors/datasets")
    organize_euvp(datasets_root)
    organize_suim(datasets_root)
    print("Done organizing datasets.")
