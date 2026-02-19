import random
import shutil

from pathlib import Path


def copy_structure_with_selected_patients(source_dir, target_dir, num_selected=2):
    source = Path(source_dir)
    target = Path(target_dir)

    # Step 1: Copy the directory structure (excluding files)
    for src_subdir in source.rglob("*"):
        if src_subdir.is_dir():
            new_subdir = target / src_subdir.relative_to(source)
            new_subdir.mkdir(parents=True, exist_ok=True)  # Create subdirectories

    # Step 2: Select N random subdirectories from "Patients"
    patients_dir = source / "Patients"
    gt_dir = source / "GT"

    patient_subdirs = [d for d in patients_dir.iterdir() if d.is_dir()]
    
    if len(patient_subdirs) < num_selected:
        raise ValueError(f"Not enough subdirectories to select {num_selected}. Found {len(patient_subdirs)}.")

    selected_subdirs = random.sample(patient_subdirs, num_selected)

    # Step 3: Copy only selected subdirectories from "Patients" and "GT"
    for subdir in selected_subdirs:
        relative_path = subdir.relative_to(source)
        
        # Copy the selected "Patients" subdirectory
        new_patient_subdir = target / relative_path
        shutil.copytree(subdir, new_patient_subdir, dirs_exist_ok=True)
        
        # Copy the corresponding "GT" subdirectory
        gt_subdir = gt_dir / subdir.name
        new_gt_subdir = target / "GT" / subdir.name
        if gt_subdir.exists():
            shutil.copytree(gt_subdir, new_gt_subdir, dirs_exist_ok=True)

# Example usage
source_directory = "/scratch/radv/share/glioseg/65_patients_clean"
target_directory = "/scratch/radv/share/glioseg/2_patients"

copy_structure_with_selected_patients(source_directory, target_directory, num_selected=2)
print(" Structure copied, and selected 'Patients' & 'GT' subdirectories transferred!")
