# MNI152_ORIENTATION = "RAS"
# MNI152_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
# MNI152_ANGLES = [0, 0, 180]
# MNI152_CENTER = (0.01626615936029907, 21.274018629494194, 10.16421050446013)

from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_information_im(im_dir: str | Path, brain_mask_dir: str | Path):

    orientation_filter = sitk.DICOMOrientImageFilter()
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()

    im_info = {}

    if isinstance(im_dir, str):
        im_dir = Path(im_dir)
    if isinstance(brain_mask_dir, str):
        brain_mask_dir = Path(brain_mask_dir)

    atlas_im = sitk.ReadImage(str(im_dir))
    brain_mask_dir = sitk.ReadImage(str(brain_mask_dir), sitk.sitkUInt8)

    atlas_size = atlas_im.GetSize()
    atlas_origin = atlas_im.GetOrigin()
    atlas_spacing = atlas_im.GetSpacing()
    im_direction = atlas_im.GetDirection()

    im_info["Size"] = atlas_size
    im_info["Origin"] = atlas_origin
    im_info["Spacing"] = atlas_spacing
    im_info["Direction"] = im_direction

    axcodes = orientation_filter.GetOrientationFromDirectionCosines(im_direction)
    im_info["Orientation"] = ''.join(axcodes)

    label_shape_filter.Execute(brain_mask_dir)
    center = label_shape_filter.GetCentroid(1)
    im_info["Center"] = center

    rot = R.from_matrix(np.array(im_direction).reshape(3, 3))
    angles = rot.as_euler('XYZ', degrees=True)
    im_info["Angles"] = angles

    return im_info

if __name__ == "__main__":

    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()

    input_dir = Path("/gpfs/work1/0/prjs0971/glioseg/data/BraTS2023_relabeled/Patients")
    output_dir = Path("/home/gesteban/glioseg/glioseg/extras")

    mni152_atlas_dir = "/gpfs/work1/0/prjs0971/glioseg/data/mni_icbm152_nlin_sym_09a"
    mni152_atlas_t1 = Path(mni152_atlas_dir) / "mni_icbm152_t1_tal_nlin_sym_09a.nii"
    mni152_atlas_t2 = Path(mni152_atlas_dir) / "mni_icbm152_t2_tal_nlin_sym_09a.nii"
    mni152_atlas_brain_mask = Path(mni152_atlas_dir) / "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

    # mni152_atlas_dir = "/home/gesteban/glioseg/glioseg/registration/atlas"
    # mni152_atlas_t1 = Path(mni152_atlas_dir) / "Atlas_mni_icbm152_nonlinear_symmetric_2009a_T1.nii.gz"
    # mni152_atlas_t2 = Path(mni152_atlas_dir) / "Atlas_mni_icbm152_nonlinear_symmetric_2009a_T2.nii.gz"
    # mni152_atlas_brain_mask = Path(mni152_atlas_dir) / "Atlas_mask.nii.gz"


    sri_24_atlas_dir = "/gpfs/work1/0/prjs0971/glioseg/data/sri24_spm8/templates"
    sri24_atlas_t1 = Path(sri_24_atlas_dir) / "T1.nii"
    sri24_atlas_t2 = Path(sri_24_atlas_dir) / "T2.nii"
    sri24_atlas_brain_mask_t1 = Path(sri_24_atlas_dir) / "T1_brain_mask.nii"
    sri24_atlas_brain_mask_t2 = Path(sri_24_atlas_dir) / "T2_brain_mask.nii"
    sri24_atlas_brain_mask_t1_im = sitk.ReadImage(str(sri24_atlas_brain_mask_t1), sitk.sitkUInt8)
    sri24_atlas_brain_mask_t2_im = sitk.ReadImage(str(sri24_atlas_brain_mask_t2), sitk.sitkUInt8)

    mni152_t1_info = get_information_im(mni152_atlas_t1, mni152_atlas_brain_mask)
    mni152_t2_info = get_information_im(mni152_atlas_t2, mni152_atlas_brain_mask)

    sri24_t1_info = get_information_im(sri24_atlas_t1, sri24_atlas_brain_mask_t1)
    sri24_t2_info = get_information_im(sri24_atlas_t2, sri24_atlas_brain_mask_t2)


    print("MNI152 T1 info:")
    for key, value in mni152_t1_info.items():
        print(f"{key}: {value}")
    print("-"*100)
    print("MNI152 T2 info:")
    for key, value in mni152_t2_info.items():
        print(f"{key}: {value}")
    print("-"*100)
    print("SRI24 T1 info:")
    for key, value in sri24_t1_info.items():
        print(f"{key}: {value}")
    print("-"*100)
    print("SRI24 T2 info:")
    for key, value in sri24_t2_info.items():
        print(f"{key}: {value}")
    print("-"*100)


    # overlap_filter.Execute(sri24_atlas_brain_mask_t1_im, sri24_atlas_brain_mask_t2_im)
    # dice_score_brain_mask_t1_t2_sri = overlap_filter.GetDiceCoefficient()
    # print(f"Dice score between SRI24 T1 and T2 brain masks: {dice_score_brain_mask_t1_t2_sri}")


    # or_image_filter = sitk.OrImageFilter()
    # combined_sri24_brain_mask = or_image_filter.Execute(sri24_atlas_brain_mask_t1_im, sri24_atlas_brain_mask_t2_im)
    # sitk.WriteImage(combined_sri24_brain_mask, str(Path(sri_24_atlas_dir) / "SRI24_combined_brain_mask.nii.gz"))

    
    # and_image_filter = sitk.AndImageFilter()
    # and_sri24_brain_mask = and_image_filter.Execute(sri24_atlas_brain_mask_t1_im, sri24_atlas_brain_mask_t2_im)
    # sitk.WriteImage(and_sri24_brain_mask, str(Path(sri_24_atlas_dir) / "SRI24_and_brain_mask.nii.gz"))

    # overlap_filter.Execute(sri24_atlas_brain_mask_t1_im, combined_sri24_brain_mask)
    # dice_score_t1_with_combined = overlap_filter.GetDiceCoefficient()
    # overlap_filter.Execute(sri24_atlas_brain_mask_t2_im, combined_sri24_brain_mask)
    # dice_score_t2_with_combined = overlap_filter.GetDiceCoefficient()
    # print(f"Dice score between SRI24 T1 brain mask and combined brain mask: {dice_score_t1_with_combined}")
    # print(f"Dice score between SRI24 T2 brain mask and combined brain mask: {dice_score_t2_with_combined}")


    # overlap_filter.Execute(sri24_atlas_brain_mask_t1_im, and_sri24_brain_mask)
    # dice_score_t1_with_and = overlap_filter.GetDiceCoefficient()
    # overlap_filter.Execute(sri24_atlas_brain_mask_t2_im, and_sri24_brain_mask)
    # dice_score_t2_with_and = overlap_filter.GetDiceCoefficient()
    # print(f"Dice score between SRI24 T1 brain mask and and brain mask: {dice_score_t1_with_and}")
    # print(f"Dice score between SRI24 T2 brain mask and and brain mask: {dice_score_t2_with_and}")


    # sri24_with_combined_info_t1 = get_information_im(sri24_atlas_t1, Path(sri_24_atlas_dir) / "SRI24_combined_brain_mask.nii.gz")
    # sri24_with_combined_info_t2 = get_information_im(sri24_atlas_t2, Path(sri_24_atlas_dir) / "SRI24_combined_brain_mask.nii.gz")

    # print("SRI24 T1 info with combined brain mask:")
    # for key, value in sri24_with_combined_info_t1.items():
    #     print(f"{key}: {value}")
    # print("-"*100)
    # print("SRI24 T2 info with combined brain mask:")
    # for key, value in sri24_with_combined_info_t2.items():
    #     print(f"{key}: {value}")


    ########

    # example_brats_case_dir = Path("/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00680-000")

    # example_brats_t1 = example_brats_case_dir / f"{example_brats_case_dir.name}-t1n.nii.gz"
    # example_brats_t1c = example_brats_case_dir / f"{example_brats_case_dir.name}-t1c.nii.gz"
    # example_brats_t2 = example_brats_case_dir / f"{example_brats_case_dir.name}-t2w.nii.gz"
    # example_brats_flair = example_brats_case_dir / f"{example_brats_case_dir.name}-t2f.nii.gz"
    # example_brats_t1_brain_mask = example_brats_case_dir / f"{example_brats_case_dir.name}-t1n_brain_mask.nii.gz"
    # example_brats_t1ce_brain_mask = example_brats_case_dir / f"{example_brats_case_dir.name}-t1c_brain_mask.nii.gz"
    # example_brats_t2_brain_mask = example_brats_case_dir / f"{example_brats_case_dir.name}-t2w_brain_mask.nii.gz"
    # example_brats_flair_brain_mask = example_brats_case_dir / f"{example_brats_case_dir.name}-t2f_brain_mask.nii.gz"

    # info_example_t1 = get_information_im(example_brats_t1, example_brats_t1_brain_mask)
    # info_example_t1ce = get_information_im(example_brats_t1c, example_brats_t1ce_brain_mask)
    # info_example_t2 = get_information_im(example_brats_t2, example_brats_t2_brain_mask)
    # info_example_flair = get_information_im(example_brats_flair, example_brats_flair_brain_mask)

    # print("Example BraTS case T1 info:")
    # for key, value in info_example_t1.items():
    #     print(f"{key}: {value}")
    # print("-"*100)

    # print("Example BraTS case T1ce info:")
    # for key, value in info_example_t1ce.items():
    #     print(f"{key}: {value}")
    # print("-"*100)

    # print("Example BraTS case T2 info:")
    # for key, value in info_example_t2.items():
    #     print(f"{key}: {value}")
    # print("-"*100)

    # print("Example BraTS case FLAIR info:")
    # for key, value in info_example_flair.items():
    #     print(f"{key}: {value}")
    # print("-"*100)
    









