import json

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import glioseg.constants as constants


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
    same_direction = all(
        abs(a - b) < 1e-6 for a, b in zip(img1.GetDirection(), img2.GetDirection())
    )

    if verbose:
        print(f"Size match:       {same_size} ({img1.GetSize()} vs {img2.GetSize()})")
        print(f"Spacing match:    {same_spacing} ({img1.GetSpacing()} vs {img2.GetSpacing()})")
        print(f"Origin match:     {same_origin} ({img1.GetOrigin()} vs {img2.GetOrigin()})")
        print(f"Direction match:  {same_direction}")

    return all([same_size, same_spacing, same_origin, same_direction])


def check_registration_status(
    input_dir: str | Path,
    output_dir: str | Path,
    atlas_t1_file: str | Path,
    atlas_t2_file: str | Path,
):

    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(atlas_t1_file, str):
        atlas_t1_file = Path(atlas_t1_file)
    if isinstance(atlas_t2_file, str):
        atlas_t2_file = Path(atlas_t2_file)

    success = 0
    partial_success = 0 
    failure = 0
    incomplete = 0

    for patient in input_dir.iterdir():

        registration_status_file = patient / "REGISTRATION" / "LOGS" / "registration_status.json"

        if registration_status_file.exists():
            
            
            registration_status_file = json.load(open(registration_status_file, "r"))
            print(registration_status_file)
            registration_status = registration_status_file["Registration status"]

            if registration_status == constants.REGISTRATION_STATUS.SUCCESS.value:
                success += 1
            
            elif registration_status == constants.REGISTRATION_STATUS.PARTIAL_SUCCESS.value:
                partial_success += 1
            
            elif registration_status == constants.REGISTRATION_STATUS.INCOMPLETE.value:
                incomplete += 1
            
            else:
                failure += 1

    print(f"Successful registrations: {success}")
    print(f"Partial successful registrations: {partial_success}")
    print(f"Incomplete registrations: {incomplete}")
    print(f"Failed registrations: {failure}")


if __name__ == "__main__":

    input_dir = Path("/gpfs/work1/0/prjs0971/glioseg/data/BraTS2023_relabeled_2/Patients")
    output_dir = Path("/home/gesteban/glioseg/glioseg/extras/results")

    sri_24_atlas_dir = "/projects/0/prjs0971/glioseg/data/sri24_spm8/templates"

    atlas_t1_file_dir = Path(sri_24_atlas_dir) / "T1_brain_mask.nii"
    atlas_t2_file_dir = Path(sri_24_atlas_dir) / "T2_brain_mask.nii"

    check_registration_status(input_dir, output_dir, atlas_t1_file_dir, atlas_t2_file_dir)
