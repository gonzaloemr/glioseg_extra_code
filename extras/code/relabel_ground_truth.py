from pathlib import Path

import SimpleITK as sitk


pat_ids = ["IM0382", "IM0385", "IM1392", "IM1444"]
root_path = Path("/scratch/radv/share/glioseg/new_run_corrected/GT")


def relabel_mask(mask_path):
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")


    backup_path = mask_path.parent / "MASK_old_labels.nii.gz"


    img = sitk.ReadImage(str(mask_path))
    sitk.WriteImage(img, str(backup_path))


    arr = sitk.GetArrayFromImage(img)
    arr[arr == 2] = 3


    updated = sitk.GetImageFromArray(arr)
    updated.CopyInformation(img) 
    sitk.WriteImage(updated, str(mask_path))

 

def main() -> None:
    for pid in pat_ids:
        mask_path = root_path / pid / "NIFTI" / "MASK.nii.gz"
        print(f"Processing {pid}")
        relabel_mask(mask_path)

 

if __name__ == "__main__":
    main()