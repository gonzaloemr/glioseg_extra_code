# Comparison of volumes from deepscan model with and without augmentation and its in full and lite mode 

# First, get the volumes path:
import os

import nibabel as nib
import numpy as np
import pandas as pd


scan_1p_path = "/projects/0/prjs0971/glioseg/data/scan20_1p"

# Pipeline full mode with and without augmentation
# full_mode_aug_path = os.path.join(scan_1p_path, "mask_tumor_full_aug.nii.gz")
# full_mode_noaug_path = os.path.join(scan_1p_path, "mask_tumor_full_noaug.nii.gz")
full_mode_aug_path = os.path.join(scan_1p_path, "full_aug_v2.nii.gz")
full_mode_noaug_path = os.path.join(scan_1p_path, "full_noaug_v2.nii.gz")

# Pipeline lite mode with and without augmentation 
# lite_mode_aug_path = os.path.join(scan_1p_path, "mask_tumor_lite_aug.nii.gz")
# lite_mode_noaug_path = os.path.join(scan_1p_path, "mask_tumor_lite_noaug.nii.gz")
lite_mode_aug_path = os.path.join(scan_1p_path, "lite_aug_v2.nii.gz")
lite_mode_noaug_path = os.path.join(scan_1p_path, "lite_noaug_v2.nii.gz")
# Load the volumes 
full_mode_aug = nib.load(full_mode_aug_path).get_fdata()
full_mode_noaug = nib.load(full_mode_noaug_path).get_fdata()
lite_mode_aug = nib.load(lite_mode_aug_path).get_fdata()
lite_mode_noaug = nib.load(lite_mode_noaug_path).get_fdata()

# print(np.array_equal(full_mode_aug, full_mode_noaug))

# Now we print the shape:

full_mode_all_copy = nib.load(os.path.join(scan_1p_path, "full_aug_v2.nii.gz")).get_fdata()
full_mode_two_copy = nib.load(os.path.join(scan_1p_path, "full_aug_twocopies.nii.gz")).get_fdata()
full_mode_one_copy = nib.load(os.path.join(scan_1p_path, "full_aug_onecopy.nii.gz")).get_fdata()
full_mode_one_copy_2 = nib.load(os.path.join(scan_1p_path, "mask_tumor_scan2020.nii.gz")).get_fdata()

print(np.array_equal(full_mode_all_copy,full_mode_two_copy))
print(np.array_equal(full_mode_all_copy,full_mode_one_copy))
print(np.array_equal(full_mode_all_copy,full_mode_one_copy_2))

# print(f"Full mode with augmentation shape: {full_mode_aug.shape}")
# print(f"Full mode without augmentation shape: {full_mode_noaug.shape}")
# print(f"Lite mode with augmentation shape: {lite_mode_aug.shape}")
# print(f"Lite mode without augmentation shape: {lite_mode_noaug.shape}")

def dice_score(pred, mask, classes):

    '''
    This function computes the Dice score for each class and averages them to obtain the overall score for a given predicted mask
    '''

    smooth = 1e-8
    dice_scores = []

    for class_id in classes:
      mask_class = (mask == class_id).astype(int)
      pred_class = (pred == class_id).astype(int)
      intersection = 2.0 * (pred_class * mask_class).sum()
      union = pred_class.sum() + mask_class.sum() +smooth
      
      dice = intersection / union
      dice_scores.append(dice)

    return dice_scores

def get_dice_score(seg, gt, labels):
    seg = np.round(seg)
    gt = np.round(gt)

    dice = []
    for label in labels:
        num = np.sum(seg[gt == label] == label) * 2.0
        den = np.sum(seg == label) + np.sum(gt == label)
        if den == 0:
            d = 1
        else:
            d = num / den
        dice.append(d)

    return dice



def compare_masks(mode_aug, mode_no_aug, labels):
    """
    Compare two masks and calculate the overlap and non-overlap voxel counts for each label.
    
    Parameters:
        mode_aug (np.ndarray): First mask (3D or 4D).
        mode_no_aug (np.ndarray): Second mask (same shape as mode_aug).
        labels (list): List of labels to compare (e.g., [1, 2, 3, 4]).

    Returns:
        dict: Dictionary with label as key and (overlap, non-overlap) counts as values.
    """
    result = {}
    for label in labels:
        mode_aug_label = (mode_aug == label).astype(int)
        mode_no_aug_label = (mode_no_aug == label).astype(int)

        total_voxels_mode_aug = np.sum(mode_aug_label)
        total_voxels_mode_no_aug = np.sum(mode_no_aug_label) 

        overlap = np.sum(np.logical_and(mode_aug_label, mode_no_aug_label))
        non_overlap_mode_aug = np.sum(np.logical_and(mode_aug_label, np.logical_not(mode_no_aug_label)))
        non_overlap_mode_no_aug = np.sum(np.logical_and(np.logical_not(mode_aug_label), mode_no_aug_label))

        result[(f"Label_{label}", 'Total voxels mode aug')] = [int(total_voxels_mode_aug)]
        result[(f"Label_{label}", 'Total voxels mode no aug')] = [int(total_voxels_mode_no_aug)]
        result[(f"Label_{label}", 'Overlap')] = [int(overlap)]
        result[(f"Label_{label}", 'Unique voxels mode aug')] = [int(non_overlap_mode_aug)]
        result[(f"Label_{label}", 'Unique voxels mode no aug')] = [int(non_overlap_mode_no_aug)]
        result[(f"Label_{label}", '% Unique voxels mode aug')] = [round(int(non_overlap_mode_aug)/total_voxels_mode_aug,3)]
        result[(f"Label_{label}", '% Unique voxels mode no aug')] = [round(int(non_overlap_mode_no_aug)/total_voxels_mode_no_aug,3)]

    return result

print(dice_score(full_mode_all_copy,full_mode_two_copy,[1,2,4]))
print(dice_score(full_mode_all_copy,full_mode_one_copy,[1,2,4]))
print(dice_score(full_mode_all_copy,full_mode_one_copy_2,[1,2,4]))
print(dice_score(full_mode_one_copy,full_mode_one_copy_2,[1,2,4]))
# comparison_full = compare_masks(full_mode_aug, full_mode_noaug, [1, 2, 4])
# comparison_lite = compare_masks(lite_mode_aug, lite_mode_noaug, [1, 2, 4])


# comparison_full = pd.DataFrame(comparison_full)
# comparison_lite = pd.DataFrame(comparison_lite)

# comparison_full.columns = pd.MultiIndex.from_tuples(comparison_full.columns)
# comparison_lite.columns = pd.MultiIndex.from_tuples(comparison_lite.columns)

# comparison_full.to_csv("full_model.csv",index=False)
# comparison_lite.to_csv("lite_model.csv",index=False)

# # Now we compute the Dice score between the 'best' version of the model and the rest of them:

# dice_full_aug_with_full_no_aug = dice_score(full_mode_noaug,full_mode_aug, [1,2,4])
# dice_full_aug_with_lite_aug = dice_score(lite_mode_aug,full_mode_aug, [1,2,4])
# dice_full_aug_with_lite_no_aug = dice_score(lite_mode_noaug,full_mode_aug, [1,2,4])
# dice_full_aug_with_full_aug = dice_score(full_mode_aug, full_mode_aug, [1,2,4])

# print(dice_full_aug_with_full_no_aug)
# print(dice_full_aug_with_lite_aug)
# print(dice_full_aug_with_lite_no_aug)
# print(dice_full_aug_with_full_aug)

# dice_full_aug_with_full_no_aug = get_dice_score(full_mode_noaug,full_mode_aug, [1,2,4])
# dice_full_aug_with_lite_aug = get_dice_score(lite_mode_aug,full_mode_aug, [1,2,4])
# dice_full_aug_with_lite_no_aug = get_dice_score(lite_mode_noaug,full_mode_aug, [1,2,4])
# dice_full_aug_with_full_aug = get_dice_score(full_mode_aug, full_mode_aug, [1,2,4])


# print(dice_full_aug_with_full_no_aug)
# print(dice_full_aug_with_lite_aug)
# print(dice_full_aug_with_lite_no_aug)
# print(dice_full_aug_with_full_aug)