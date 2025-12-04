import shutil

from pathlib import Path

import itk

import glioseg.constants as constants
import glioseg.IO.utils as IO_utils


def transform_mask_to_atlas_space(gt_root: Path) -> None:
    """
    Transforms MASK.nii.gz from T1GD space to ATLAS space for each patient under `gt_root`.

    Args:
        gt_root (Path): Path to GT folder, e.g. /scratch/radv/share/glioseg/GT_Vera/GT
    """
    for patient_folder in sorted(gt_root.iterdir()):
        if not patient_folder.is_dir():
            continue

        patient_id = patient_folder.name
        nifti_input = patient_folder / "NIFTI" / "MASK.nii.gz"
        if not nifti_input.exists():
            print(f"Skipping {patient_id}: MASK.nii.gz not found.")
            continue

        # Define paths
        patient_root = gt_root.parent / "Patients" / patient_id
        tmp_seg_folder = patient_root / "SEGMENTATIONS" / "NIFTI"
        tmp_seg_folder.mkdir(parents=True, exist_ok=True)
        tmp_mask_path = tmp_seg_folder / "MASK.nii.gz"

        # Copy MASK to expected location
        shutil.copy(nifti_input, tmp_mask_path)

        # Define output path in ATLAS space
        output_mask_atlas = patient_folder / "ATLAS" / "MASK.nii.gz"
        output_mask_atlas.parent.mkdir(exist_ok=True, parents=True)

        try:
            transform_mask_with_nearest_neighbor(
                patient_folder=patient_root,
                input_image_path=tmp_mask_path,
                output_image_path=output_mask_atlas,
                modality="T1GD",
            )
            print(f"Transformed {patient_id} MASK to ATLAS space.")
        except Exception as e:
                print(f"Error processing {patient_id}: {e}")

        # Cleanup temporary file
        try:
            tmp_mask_path.unlink()
        except Exception:
            pass


def transform_mask_with_nearest_neighbor(
    patient_folder: Path, input_image_path: Path, output_image_path: Path, modality: str
) -> None:
    """
    Transform a segmentation mask from T1GD to ATLAS space using nearest neighbor interpolation.

    Args:
        patient_folder (Path): Path to patient root folder (under GT_Vera/Patients)
        input_image_path (Path): Path to MASK.nii.gz
        output_image_path (Path): Where to save transformed mask
        modality (str): Modality (e.g., T1GD)
    """
    registration_status_file = patient_folder.joinpath(
        constants.REGISTRATION_FOLDER_NAME,
        constants.LOG_FOLDER_NAME,
        constants.REGISTRATION_STATUS_FILE,
    ).with_suffix(constants.JSON_EXTENSION)

    registration_status_data = IO_utils.load_json(registration_status_file)
    registration_status = constants.REGISTRATION_STATUS(
        registration_status_data[constants.REGISTRATION_STATUS_KEYWORD]
    )

    parameter_folder = patient_folder.joinpath(constants.ITK_PARAMETER_FOLDER_NAME)


    registration_status_keys = list(registration_status_data.keys())

    pairwise_step_2_failed = False 

    if constants.PAIRWISE_REGISTERED_SCANS_FAILED_KEYWORD in registration_status_keys and registration_status_data[constants.PAIRWISE_REGISTERED_SCANS_FAILED_KEYWORD]:
        pairwise_step_2_failed = True
    
    transforms = {
        "nifti_to_reoriented": IO_utils.load_json(
            parameter_folder / constants.JSON_REORIENTED_FILE.format(modality=modality)
        ),
        "reoriented_to_resampled": itk.transformread(
            parameter_folder / constants.ITK_RESAMPLING_TRANSFORM_PARAMETER_FILE.format(modality=modality)
        ),
        "resampled_to_pairwise_brain_mask": itk.transformread(
            parameter_folder / constants.ITK_RESAMPLED_TO_PAIRWISE_BRAIN_MASKS_TRANSFORM_PARAMETER_FILE.format(modality=modality)
        ),
        "pairwise_brain_mask_to_pairwise": itk.transformread(
            parameter_folder / constants.ITK_PAIRWISE_BRAIN_MASKS_TO_PAIRWISE_TRANSFORM_PARAMETER_FILE.format(modality=modality)
        ),
        "pairwise_to_atlas_brain_mask": itk.transformread(
            parameter_folder / constants.ITK_PAIRWISE_TO_ATLAS_BRAIN_MASKS_TRANSFORM_PARAMETER_FILE
        ),
    }
    

    if registration_status is constants.REGISTRATION_STATUS.SUCCESS:
        transforms["multi_image_or_pairwise_atlas_brain_mask_to_atlas"] = itk.transformread(
            parameter_folder / constants.ITK_ATLAS_BRAIN_MASKS_TO_ATLAS_PARAMETER_FILE
        )

    # Load reference images
    ref_folder = patient_folder / constants.REGISTRATION_FOLDER_NAME

    reference_images = {
        "reoriented": itk.imread((patient_folder / constants.REORIENT_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
        "resampled": itk.imread((patient_folder / constants.RESAMPLE_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
        "pairwise_brain_mask": itk.imread((ref_folder / constants.PAIRWISE_REGISTRATION_BRAIN_MASK_ONLY_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
        "pairwise": itk.imread((ref_folder / constants.PAIRWISE_REGISTRATION_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
        "atlas_brain_mask": itk.imread((ref_folder / constants.ATLAS_REGISTRATION_BRAIN_MASK_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
        "atlas": itk.imread((ref_folder / constants.ATLAS_REGISTRATION_FOLDER_NAME / modality).with_suffix(constants.NIFTI_EXTENSION)),
    }

    if pairwise_step_2_failed:
        print(f"Warning: Pairwise registration (step 2) failed for {patient_folder.name}")
        reference_images.pop("pairwise")

    # Apply transformations with nearest neighbor interpolation
    image = itk.imread(input_image_path, itk.UC)  # use unsigned char for labels
    image = apply_reorientation(image, transforms["nifti_to_reoriented"], reference_images["reoriented"])
    image = resample_image(image, transforms["reoriented_to_resampled"], reference_images["resampled"], interpolator="nn")
    image = resample_image(image, transforms["resampled_to_pairwise_brain_mask"], reference_images["pairwise_brain_mask"], interpolator="nn")
    if not pairwise_step_2_failed:
        image = resample_image(image, transforms["pairwise_brain_mask_to_pairwise"], reference_images["pairwise"], interpolator="nn")
    image = resample_image(image, transforms["pairwise_to_atlas_brain_mask"], reference_images["atlas_brain_mask"], interpolator="nn")

    if registration_status is constants.REGISTRATION_STATUS.SUCCESS:
        image = resample_image(image, transforms["multi_image_or_pairwise_atlas_brain_mask_to_atlas"], reference_images["atlas"], interpolator="nn")

    itk.imwrite(image, output_image_path)


def apply_reorientation(image: itk.Image, reorientation_parameters: dict, reference_image: itk.Image) -> itk.Image:
    permutation_filter = itk.PermuteAxesImageFilter.New(Input=image)
    permutation_filter.SetOrder(reorientation_parameters["permutation"])
    permutation_filter.Update()
    image = permutation_filter.GetOutput()

    flip_filter = itk.FlipImageFilter.New(Input=image)
    flip_filter.SetFlipAxes(reorientation_parameters["axesflip"])
    flip_filter.SetFlipAboutOrigin(False)
    flip_filter.Update()
    image = flip_filter.GetOutput()

    rotations = reorientation_parameters["position_rotations"]
    center = get_image_center(image)
    euler = itk.Euler3DTransform[itk.D].New()
    euler.SetRotation(*rotations)
    euler.SetCenter(center)

    return resample_image(image, euler, reference_image, interpolator="nn")


def get_image_center(image: itk.Image):
    size = image.GetLargestPossibleRegion().GetSize()
    return image.TransformIndexToPhysicalPoint([
        int(size[0] / 2),
        int(size[1] / 2),
        int(size[2] / 2),
    ])


def resample_image(image: itk.Image, transform, reference_image: itk.Image, interpolator: str = "linear") -> itk.Image:
    if interpolator == "nn":
        interp = itk.NearestNeighborInterpolateImageFunction.New(image)
    else:
        interp = itk.LinearInterpolateImageFunction.New(image)

    return itk.resample_image_filter(
        image,
        transform=transform,
        use_reference_image=True,
        reference_image=reference_image,
        interpolator=interp,
    )


if __name__ == "__main__":
    transform_mask_to_atlas_space(Path("/scratch/radv/share/glioseg/new_run_corrected/GT/"))
