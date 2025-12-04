import SimpleITK as sitk
from pathlib import Path
import numpy as np

def same_physical_space(img1: sitk.Image, img2: sitk.Image, verbose: bool = False) -> bool:
    """
    Check if two SimpleITK images share the same physical space 
    (size, spacing, origin, direction).

    Args:
        img1 (sitk.Image): First image.
        img2 (sitk.Image): Second image.
        verbose (bool): If True, prints differences.

    Returns:
        bool: True if all spatial properties match, False otherwise.
    """
    same_size = img1.GetSize() == img2.GetSize()
    same_spacing = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetSpacing(), img2.GetSpacing()))
    same_origin = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetOrigin(), img2.GetOrigin()))
    same_direction = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetDirection(), img2.GetDirection()))

    if verbose:
        print(f"Size match:       {same_size} ({img1.GetSize()} vs {img2.GetSize()})")
        print(f"Spacing match:    {same_spacing} ({img1.GetSpacing()} vs {img2.GetSpacing()})")
        print(f"Origin match:     {same_origin} ({img1.GetOrigin()} vs {img2.GetOrigin()})")
        print(f"Direction match:  {same_direction}")

    return all([same_size, same_spacing, same_origin, same_direction])


def check_BraTS_physical_space(input_dir: str | Path, sri_24_atlas_dir: str | Path, modality_maps: dict):

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    if not isinstance(sri_24_atlas_dir, Path):
        sri_24_atlas_dir = Path(sri_24_atlas_dir)

    for patient in input_dir.iterdir():
        
        patient_name = patient.name 
        
        for modality_brats, atlas_template in modality_maps.items():

            modality_brats_file = patient / f"{patient_name}-{modality_brats}.nii.gz"
            atlas_template_file = sri_24_atlas_dir / f"{atlas_template}.nii"

            modality_brats_im = sitk.ReadImage(str(modality_brats_file))
            atlas_template_im = sitk.ReadImage(str(atlas_template_file))

            print(f"Checking physical space for Patient: {patient_name}, Modality: {modality_brats}")
            same_space = same_physical_space(modality_brats_im, atlas_template_im, verbose=True)
            print(f"Same physical space: {same_space}\n")

if __name__ == "__main__":
    
    
    input_dir = "/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/reoriented"
    sri_24_atlas_dir = "/gpfs/work1/0/prjs0971/glioseg/data/sri24_spm8/templates"
    modality_maps = {
        "t1n": "T1",
        "t1c": "T1",
        "t2w": "T2",
        "t2f": "T2"
    }

    check_BraTS_physical_space(input_dir, sri_24_atlas_dir, modality_maps)


