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

    registration_summary = []
    lens_successful_cases = []
    successful_cases = {2: [], 3: [], 4: [], 5: []}

    for patient in input_dir.iterdir():

        registration_status_file = patient / "REGISTRATION" / "LOGS" / "registration_status.json"

        if registration_status_file.exists():
            registration_status_file = json.load(open(registration_status_file, "r"))
            registration_status = registration_status_file["Registration status"]

            # registration_summary.append({
            #     "Patient ID": patient.name,
            #     "Registration finished": True,
            #     "Registration status": registration_status,
            #     "Status summary": False if "failed" in registration_status.lower() else True})

            # print(registration_status == constants.REGISTRATION_STATUS.SUCCESS)
            if registration_status == constants.REGISTRATION_STATUS.SUCCESS.value:
                status_summary = "SUCCESS"
                lens_successful_cases.append(len(registration_status_file.keys()))
                successful_cases[len(registration_status_file.keys())].append(
                    registration_status_file
                )

                if len(registration_status_file.keys()) == 2:
                    print(patient.name)

            elif registration_status == constants.REGISTRATION_STATUS.INCOMPLETE.value:
                status_summary = "INCOMPLETE"
            else:
                status_summary = "FAILED"

            registration_summary.append(
                {
                    "Patient ID": patient.name,
                    "Registration finished": True,
                    "Registration status": registration_status,
                    "Status summary": status_summary,
                }
            )

            # if "failed" in registration_status.lower():
            #     print(registration_status_file)
            #     print(f"{patient.name}")

            # t1_registered_file = patient / "REGISTRATION" / "ATLAS_SRI24" / "T1.nii"
            # t1_gd_registered_file = patient / "REGISTRATION" / "ATLAS_SRI24" / "T1GD.nii"
            # t2_registered_file = patient / "REGISTRATION" / "ATLAS_SRI24" / "T2.nii"
            # flair_registered_file = patient / "REGISTRATION" / "ATLAS_SRI24" / "FLAIR.nii"

            # t1_registered_im = sitk.ReadImage(str(t1_registered_file))
            # t1_gd_registered_im = sitk.ReadImage(str(t1_gd_registered_file))
            # t2_registered_im = sitk.ReadImage(str(t2_registered_file))
            # flair_registered_im = sitk.ReadImage(str(flair_registered_file))
            # atlas_t1_im = sitk.ReadImage(str(atlas_t1_file))
            # atlas_t2_im = sitk.ReadImage(str(atlas_t2_file))

            # # if "failed" in registration_status.lower():
            # t1_check = same_physical_space(t1_registered_im, atlas_t1_im)
            # t1_gd_check = same_physical_space(t1_gd_registered_im, atlas_t1_im)
            # t2_check = same_physical_space(t2_registered_im, atlas_t2_im)
            # flair_check = same_physical_space(flair_registered_im, atlas_t2_im)

            # if all([t1_check, t1_gd_check, t2_check, flair_check]):
            #     print(f"[INFO] Patient {patient.name}: Registration matches the SRI24 atlas space")
            # else:
            #     print(f"[WARNING] Patient {patient.name}: Registration does NOT match the SRI24 atlas space")

        else:
            registration_summary.append(
                {
                    "Patient ID": patient.name,
                    "Registration finished": False,
                    "Registration status": "N/A",
                    "Status summary": "N/A",
                }
            )

    # print(np.unique(lens_successful_cases, return_counts=True))

    # print(successful_cases)

    for num_key, dict_list in successful_cases.items():

        print(f"For {num_key}:")

        all_equal = all(d == dict_list[0] for d in dict_list)
        if all_equal:
            print(all_equal)
        else:
            unique_dicts = [json.dumps(d, sort_keys=True) for d in dict_list]
            unique_dicts = set(unique_dicts)
            print(unique_dicts)

    registration_summary_df = pd.DataFrame(registration_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    registration_summary_df.to_csv(output_dir / "registration_summary.csv", index=False)

    # Print how many failed and successful registrations
    total_patients = len(registration_summary_df)
    successful_registrations = len(
        registration_summary_df[registration_summary_df["Status summary"] == "SUCCESS"]
    )
    incomplete_registrations = len(
        registration_summary_df[registration_summary_df["Status summary"] == "INCOMPLETE"]
    )
    failed_registrations = len(
        registration_summary_df[registration_summary_df["Status summary"] == "FAILED"]
    )

    print(f"Total patients: {total_patients}")
    print(f"Successful registrations: {successful_registrations}")
    print(f"Incomplete registrations: {incomplete_registrations}")
    print(f"Failed registrations: {failed_registrations}")

    print(np.unique(lens_successful_cases, return_counts=True))


if __name__ == "__main__":

    input_dir = Path("/projects/0/prjs1727/glioseg/data/BraTS2023_relabeled/Patients")
    output_dir = Path("/home/gesteban/glioseg/glioseg/extras")

    sri_24_atlas_dir = "/projects/0/prjs0971/glioseg/data/sri24_spm8/templates"

    atlas_t1_file_dir = Path(sri_24_atlas_dir) / "T1_brain_mask.nii"
    atlas_t2_file_dir = Path(sri_24_atlas_dir) / "T2_brain_mask.nii"

    check_registration_status(input_dir, output_dir, atlas_t1_file_dir, atlas_t2_file_dir)
