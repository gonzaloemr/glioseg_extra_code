import os

import nibabel as nib
import numpy as np


# Load the data:

# General path of the data: 

data_path = "/projects/0/prjs0971/glioseg/data/"

# Mask 1-2-4 
mask_124 = nib.load(os.path.join(data_path, "mask_tumor_scan2020_1_2_4.nii.gz" )).get_fdata()
# Mask 1-2
mask_12 = nib.load(os.path.join(data_path, "mask_tumor_scan2020_1_4.nii.gz" )).get_fdata()
# Mask 4
mask_4 = nib.load(os.path.join(data_path, "mask_tumor_scan2020_4.nii.gz" )).get_fdata()

# Mask lite 
mask_lite = nib.load(os.path.join(data_path, "mask_tumor_scan2020_lite.nii.gz" )).get_fdata()

# Mask full  
mask_full = nib.load(os.path.join(data_path, "mask_tumor_scan2020_full.nii.gz" )).get_fdata()


def compare_masks(mask1, mask2, labels):
    """
    Compare two masks and calculate the overlap and non-overlap voxel counts for each label.
    
    Parameters:
        mask1 (np.ndarray): First mask (3D or 4D).
        mask2 (np.ndarray): Second mask (same shape as mask1).
        labels (list): List of labels to compare (e.g., [1, 2, 3, 4]).

    Returns:
        dict: Dictionary with label as key and (overlap, non-overlap) counts as values.
    """
    result = {}
    for label in labels:
        mask1_label = (mask1 == label)
        mask2_label = (mask2 == label)
        
        overlap = np.logical_and(mask1_label, mask2_label).sum()

        non_overlap_mask1 = mask1_label.sum() - overlap
        non_overlap_mask2 = mask2_label.sum() - overlap

        result[label] = {'overlap': overlap, 'non_overlap_mask1': non_overlap_mask1, 'non_overlap_mask2': non_overlap_mask2}
    
    return result

comparison_lite_full = compare_masks(mask_lite, mask_full, [1, 2, 4])
print(comparison_lite_full)
