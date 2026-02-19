from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def compare_arrays(ensemble_segmentation_file, ensemble_postprocessed_file, ensemble_postprocessed_cc_file):
    
    # Read the images and convert them to arrays
    ensemble_segmentation = sitk.ReadImage(str(ensemble_segmentation_file))
    ensemble_segmentation_array = sitk.GetArrayFromImage(ensemble_segmentation)
    ensemble_postprocessed = sitk.ReadImage(str(ensemble_postprocessed_file))
    ensemble_postprocessed_array = sitk.GetArrayFromImage(ensemble_postprocessed)
    ensemble_postprocessed_cc = sitk.ReadImage(str(ensemble_postprocessed_cc_file))
    ensemble_postprocessed_cc_array = sitk.GetArrayFromImage(ensemble_postprocessed_cc)

    # Total number of voxels in the arrays
    total_voxels = ensemble_segmentation_array.size

    # Compare arrays pairwise
    def compare_two_arrays(array1, array2):
        diff = np.sum(array1 != array2)  # Count the number of different voxels
        return diff

    # Compare ensemble_segmentation with ensemble_postprocessed
    diff_segmentation_postprocessed = compare_two_arrays(ensemble_segmentation_array, ensemble_postprocessed_array)
    
    # Compare ensemble_postprocessed with ensemble_postprocessed_cc
    diff_segmentation_postprocessed_cc = compare_two_arrays(ensemble_segmentation_array, ensemble_postprocessed_cc_array)

    # Print differences
    print(f"Difference between ensemble_segmentation and ensemble_postprocessed: {diff_segmentation_postprocessed}/{total_voxels} voxels")
    print(f"Difference between ensemble_postprocessed and ensemble_postprocessed_cc: {diff_segmentation_postprocessed_cc}/{total_voxels} voxels")
    
    return diff_segmentation_postprocessed, diff_segmentation_postprocessed_cc

def compare_arrays_all_patients(patients_dir):

    segmentation_postprocessed = []
    postprocessed_postprocessed_cc = []

    for patient in patients_dir.iterdir():

        patient_name = patient.name
        segmentations_dir = patient.joinpath("SEGMENTATIONS/ATLAS")
        ensemble_segmentation_file = segmentations_dir.joinpath("mask_tumor_ensemble.nii.gz")
        ensemble_postprocessed_file = segmentations_dir.joinpath("mask_tumor_ensemble_postprocessed.nii.gz")
        ensemble_postprocessed_cc_file = segmentations_dir.joinpath("mask_tumor_ensemble_cc.nii.gz")

        print(f"Analysis for patient {patient_name} ")
        diff_segmentation_postprocessed, diff_postprocessed_postprocessed_cc = compare_arrays(ensemble_segmentation_file, ensemble_postprocessed_file, ensemble_postprocessed_cc_file)
        segmentation_postprocessed.append(diff_segmentation_postprocessed)
        postprocessed_postprocessed_cc.append(diff_postprocessed_postprocessed_cc)
        print('-'*100)
    
    print(f"Mean difference between ensemble and postprocessed {np.mean(segmentation_postprocessed)}")
    print(f"Mean difference between postprocessed and postprocessed with non connected components {np.mean(postprocessed_postprocessed_cc)}")


if __name__ == "__main__":
    
    patients_dir = Path("/scratch/radv/share/glioseg/Patients_60_clean/Patients/")
    compare_arrays_all_patients(patients_dir)