from pathlib import Path


input_dir = Path("/scratch/radv/share/glioseg/GT_Vera/Patients")

for patient in input_dir.iterdir():
    segmentations_dir = patient / "SEGMENTATIONS" / "ATLAS"

    for seg_file in segmentations_dir.iterdir():
        
        seg_file_components = seg_file.name.split(".nii.gz")
        
        if seg_file_components[0].endswith("__8"):
            new_name = seg_file_components[0].replace("__8", "") + ".nii.gz"
            new_path = segmentations_dir / new_name
            seg_file.rename(new_path)
            print(f"Renamed {seg_file} to {new_path}")