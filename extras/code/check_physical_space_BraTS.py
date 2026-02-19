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

            if not same_physical_space(modality_brats_im, atlas_template_im, verbose=True):
                # print(f"Mismatch in physical space for patient {patient_name}, modality {modality_brats}")
                # modality_brats_im.CopyInformation(atlas_template_im)
                modality_brats_file_fixed = patient / f"{patient_name}-{modality_brats}_fixed.nii.gz"
                # sitk.WriteImage(modality_brats_im, str(modality_brats_file_fixed))
                if modality_brats_file_fixed.exists():
                    modality_brats_file_fixed.unlink()
            # print(f"Checking patient {patient_name}, modality {modality_brats} against atlas template {atlas_template}...")

            # print(modality_brats_im.GetSize(), modality_brats_im.GetSpacing(), modality_brats_im.GetOrigin(), modality_brats_im.GetDirection())
            # print(atlas_template_im.GetSize(), atlas_template_im.GetSpacing(), atlas_template_im.GetOrigin(), atlas_template_im.GetDirection())

def centroid(mask):
    """Compute physical centroid of a binary mask."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    return np.array(stats.GetCentroid(1))  # label 1 = brain

def dice_coefficient(mask1, mask2):
    """Compute Dice coefficient between two binary masks."""
    f = sitk.LabelOverlapMeasuresImageFilter()
    f.Execute(sitk.Cast(mask1, sitk.sitkUInt8), sitk.Cast(mask2, sitk.sitkUInt8))
    return f.GetDiceCoefficient()


if __name__ == "__main__":
    
    
    input_dir = "/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    sri_24_atlas_dir = "/gpfs/work1/0/prjs0971/glioseg/data/sri24_spm8/templates"
    modality_maps = {
        "t1n": "T1",
        "t1c": "T1",
        "t2w": "T2",
        "t2f": "T2"
    }

    check_BraTS_physical_space(input_dir, sri_24_atlas_dir, modality_maps)

    # example_t1_file = Path(input_dir) / "BraTS-GLI-00530-000" / "BraTS-GLI-00530-000-t1n_brain_mask.nii.gz"
    # sri24_atlas_t1_file = Path(sri_24_atlas_dir) / "T1_brain_mask.nii"

    # example_t1_im = sitk.ReadImage(str(example_t1_file))
    # sri24_atlas_t1_im = sitk.ReadImage(str(sri24_atlas_t1_file))
    # t1_im_copy_info = sitk.Image(example_t1_im)
    # t1_im_copy_info.CopyInformation(sri24_atlas_t1_im)

    # example_t1_centroid = centroid(example_t1_im)
    # sri24_atlas_t1_centroid = centroid(sri24_atlas_t1_im)
    # t1_im_copy_centroid = centroid(t1_im_copy_info)

    # centroid_dist = np.linalg.norm(t1_im_copy_centroid - sri24_atlas_t1_centroid)

    # dice = dice_coefficient(t1_im_copy_info, sri24_atlas_t1_im)

    # print(f"Example T1 centroid: {example_t1_centroid}")
    # print(f"SRI24 atlas T1 centroid: {sri24_atlas_t1_centroid}")
    # print(f"T1 image after CopyInformation centroid: {t1_im_copy_centroid}")

    # print(f"Centroid distance after CopyInformation: {centroid_dist:.4f} mm")
    # print(f"Dice coefficient after CopyInformation: {dice:.4f}")

