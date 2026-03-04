from pathlib import Path

import ants


def transform_to_nifti_space(
    input_dir: str | Path,
    gt_dir: str | Path,
) -> None:

    input_dir = Path(input_dir) if not isinstance(input_dir, Path) else input_dir
    gt_dir = Path(gt_dir) if not isinstance(gt_dir, Path) else gt_dir

    for patient in input_dir.iterdir():

        segmentation_relabeled_file = patient / f"{patient.name}_brainles" / "raw_bet_mni152" / "MASK_best_GS_relabelled.nii.gz"
        transform_file = str(patient / f"{patient.name}_brainles" / "transformations_mni152" / "t1c" / "2_M_atlas__t1c.mat")
        
        nifti_gt = ants.image_read(
            str(gt_dir.joinpath(patient.name, "NIFTI", "MASK.nii.gz"))
        )

        seg_im = ants.image_read(str(segmentation_relabeled_file))
        seg_conv = ants.apply_transforms(
            fixed=nifti_gt,
            moving=seg_im,
            transformlist=transform_file,
            interpolator="genericLabel",
            whichtoinvert=[True],
        )

        # Save the transformed image
        ants.image_write(
            seg_conv,
            filename=str(
                patient / f"{patient.name}_brainles" / "raw_bet_mni152" / "MASK_best_GS_relabelled_nifti.nii.gz"
            ),
        )

if __name__ == "__main__":

    input_dir = Path("/scratch/radv/share/glioseg/skull_stripped_scans_2/patients")
    gt_dir = Path("/scratch/radv/share/glioseg/new_run_corrected/GT/")

    transform_to_nifti_space(input_dir, gt_dir)