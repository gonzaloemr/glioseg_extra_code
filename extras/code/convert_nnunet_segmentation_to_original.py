from __future__ import annotations

import sys

from pathlib import Path

import itk
import numpy as np

from joblib import Parallel
from joblib import delayed

import glioseg.constants as constants
import glioseg.IO.config as configIO
import glioseg.IO.utils as IO_utils


def convert_nnunet_masks_all(config, input_folder, output_folder):
    config = IO_utils.load_config(config)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(input_folder.glob("*.nii.gz"))

    n_pats = len(nii_files)
    n_cpu = IO_utils.get_number_of_threads()
    n_jobs = min(n_cpu, n_pats)

    print(f"Starting {n_jobs} parallel jobs for {n_pats} patients using {n_cpu} available cores.\n")

    Parallel(n_jobs=n_jobs, backend=constants.MULTIPROCESSING_BACKEND)(
        delayed(process_single_mask)(config, nii_file, output_folder)
        for nii_file in nii_files
    )

def process_single_mask(config, nii_file, output_folder):
    patient_id = nii_file.name.replace(".nii.gz", "")
    print(f"Processing {patient_id}")

    patient_folder = config.input_data_path.joinpath(patient_id)
    if not patient_folder.exists():
        print(f"{patient_id} folder not found, skipping.")
        return

    mask_atlas = itk.imread(nii_file).astype(itk.UC)

    for modality in config.get_all_modalities:
        try:
            transformed_masks = transform_mask_to_other_frames(
                patient_folder, mask_atlas, modality
            )

            output_path = output_folder / f"{patient_id}_{modality}.nii.gz"
            itk.imwrite(transformed_masks["nifti"], output_path)
            print(f"Conversion of {patient_id} succesful")
        except Exception as e:
            print(f"Conversion of {patient_id} failed for modality {modality}: {e}")

def transform_mask_to_other_frames(
    patient_folder: Path, mask_atlas: itk.Image, modality: str
) -> dict:
    """Transforms the given mask from the atlas space to various other spaces (pairwise, resampled, reoriented, nifti).

    Args:
        patient_folder (Path): Path to the patient folder.
        mask_atlas (itk.Image): Mask image in the atlas space.
        modality (str): Modality name used for identifying the correct transforms.

    Returns:
        dict: Dictionary containing the transformed masks for different spaces.
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
    target_modality = registration_status_data.get(constants.TARGET_MODALITY_KEYWORD)
    parameter_folder = patient_folder.joinpath(constants.ITK_PARAMETER_FOLDER_NAME)

    advanced_mri_incomplete = False
    advanced_mri_skip_fail = False

    is_extra_modality = modality not in constants.MODALITIES
    if is_extra_modality:
        status_key = constants.ADVANCED_MRI_REGISTRATION_STATUS_KEYWORD.format(modality=modality)
        status_str = registration_status_data.get(status_key)
        registration_status_advanced_mri = constants.ADVANCED_MRI_REGISTRATION_STATUS(status_str)

        if (
            registration_status_advanced_mri
            is constants.ADVANCED_MRI_REGISTRATION_STATUS.INCOMPLETE
        ):
            advanced_mri_incomplete = True
        elif (
            registration_status_advanced_mri is constants.ADVANCED_MRI_REGISTRATION_STATUS.SKIPPED
            or registration_status_advanced_mri
            is constants.ADVANCED_MRI_REGISTRATION_STATUS.FAILURE
        ):
            advanced_mri_skip_fail = True

        advanced_mri_preregistration_status_file = patient_folder.joinpath(
            constants.REGISTRATION_FOLDER_NAME,
            constants.LOG_FOLDER_NAME,
            constants.ADVANCED_MRI_PREREGISTRATION_STATUS_FILE,
        ).with_suffix(constants.JSON_EXTENSION)
        advanced_mri_preregistration_status_data = IO_utils.load_json(
            advanced_mri_preregistration_status_file
        )
        advanced_mri_preregistration_status = constants.ADVANCED_MRI_PREREGISTRATION_STATUS(
            advanced_mri_preregistration_status_data[
                constants.ADVANCED_MRI_PREREGISTRATION_STATUS_KEYWORD.format(modality=modality)
            ]
        )

    # Load the transforms
    # Loading everything in reverse order, so from atlas back to original
    transforms = {}

    # Step 1: Atlas --> atlas brain masks only
    # if atlas registration (final step) was successful, load that transform
    if registration_status is constants.REGISTRATION_STATUS.SUCCESS:
        transforms["atlas_to_atlas_brain_mask"] = load_inverse_transform(
            parameter_folder.joinpath(constants.ITK_ATLAS_BRAIN_MASKS_TO_ATLAS_PARAMETER_FILE)
        )

    # Step 2: Atlas brain masks only --> pairwise
    transforms["atlas_brain_mask_to_pairwise"] = load_inverse_transform(
        parameter_folder.joinpath(
            constants.ITK_PAIRWISE_TO_ATLAS_BRAIN_MASKS_TRANSFORM_PARAMETER_FILE
        )
    )

    # Step 3: Pairwise --> pairwise brain masks only
    if not (
        modality == target_modality
        or advanced_mri_incomplete
        or advanced_mri_skip_fail
        or registration_status_data.get(constants.PAIRWISE_REGISTERED_SCANS_FAILED_KEYWORD, False)
    ):
        transforms["pairwise_to_pairwise_brain_mask"] = load_inverse_transform(
            parameter_folder.joinpath(
                constants.ITK_PAIRWISE_BRAIN_MASKS_TO_PAIRWISE_TRANSFORM_PARAMETER_FILE.format(
                    modality=modality
                ),
            )
        )

    # Step 4: Pairwise brain masks only --> resampled
    if not (modality == target_modality or advanced_mri_skip_fail):
        transforms["pairwise_brain_masks_to_resampled"] = load_inverse_transform(
            parameter_folder.joinpath(
                constants.ITK_RESAMPLED_TO_PAIRWISE_BRAIN_MASKS_TRANSFORM_PARAMETER_FILE.format(
                    modality=modality
                ),
            )
        )

    # Step 5: Resampled --> reoriented
    transforms["resampled_to_reoriented"] = load_inverse_transform(
        parameter_folder.joinpath(
            constants.ITK_RESAMPLING_TRANSFORM_PARAMETER_FILE.format(modality=modality),
        )
    )

    # Step 6: Reoriented --> nifti
    transforms["reoriented_to_nifti"] = load_inverse_reorientation_parameters(
        parameter_folder.joinpath(constants.JSON_REORIENTED_FILE.format(modality=modality))
    )

    # Load the images as references
    reference_images = {}
    reference_images["atlas_brain_mask"] = itk.imread(
        patient_folder.joinpath(
            constants.REGISTRATION_FOLDER_NAME,
            constants.ATLAS_REGISTRATION_BRAIN_MASK_FOLDER_NAME,
            modality,
        ).with_suffix(constants.NIFTI_EXTENSION)
    )
    reference_images["pairwise"] = itk.imread(
        patient_folder.joinpath(
            constants.REGISTRATION_FOLDER_NAME,
            constants.PAIRWISE_REGISTRATION_FOLDER_NAME,
            modality,
        ).with_suffix(constants.NIFTI_EXTENSION)
    )
    reference_images["pairwise_brain_mask"] = itk.imread(
        patient_folder.joinpath(
            constants.REGISTRATION_FOLDER_NAME,
            constants.PAIRWISE_REGISTRATION_BRAIN_MASK_ONLY_FOLDER_NAME,
            modality,
        ).with_suffix(constants.NIFTI_EXTENSION)
    )
    reference_images["resampled"] = itk.imread(
        patient_folder.joinpath(constants.RESAMPLE_FOLDER_NAME, modality).with_suffix(
            constants.NIFTI_EXTENSION
        )
    )
    reference_images["reoriented"] = itk.imread(
        patient_folder.joinpath(constants.REORIENT_FOLDER_NAME, modality).with_suffix(
            constants.NIFTI_EXTENSION
        )
    )
    if (
        is_extra_modality
        and advanced_mri_preregistration_status
        == constants.ADVANCED_MRI_PREREGISTRATION_STATUS.SUCCESS
    ):
        reference_images["nifti"] = itk.imread(
            patient_folder.joinpath(
                constants.REGISTRATION_FOLDER_NAME,
                constants.ADVANCED_MRI_PREREGISTRATION_FOLDER_NAME,
                modality,
            ).with_suffix(constants.NIFTI_EXTENSION)
        )
    else:
        reference_images["nifti"] = itk.imread(
            patient_folder.joinpath(constants.NIFTI_FOLDER_NAME, modality).with_suffix(
                constants.NIFTI_EXTENSION
            )
        )

    # Apply the transformations
    transformed_masks = {}
    transformed_masks["atlas"] = mask_atlas

    # Step 1: Atlas --> atlas brain masks only
    # if atlas registration (final step) was successful, load that transform
    # else, copy the masks from "atlas" to "atlas_brain_masks"
    if registration_status is constants.REGISTRATION_STATUS.SUCCESS:
        transformed_masks["atlas_brain_masks"] = resample_image(
            transformed_masks["atlas"],
            transforms["atlas_to_atlas_brain_mask"],
            reference_images["atlas_brain_mask"],
        )
    else:
        transformed_masks["atlas_brain_masks"] = transformed_masks["atlas"]

    # Step 2: Atlas brain masks only --> pairwise
    transformed_masks["pairwise"] = resample_image(
        transformed_masks["atlas_brain_masks"],
        transforms["atlas_brain_mask_to_pairwise"],
        reference_images["pairwise"],
    )

    # Step 3: Pairwise --> pairwise brain masks only
    # Skip the transform for target modality (identity) or if pairwise failed
    if (
        modality == target_modality
        or advanced_mri_incomplete
        or advanced_mri_skip_fail
        or registration_status_data.get(constants.PAIRWISE_REGISTERED_SCANS_FAILED_KEYWORD, False)
    ):
        transformed_masks["pairwise_brain_mask"] = transformed_masks["pairwise"]
    else:
        transformed_masks["pairwise_brain_mask"] = resample_image(
            transformed_masks["pairwise"],
            transforms["pairwise_to_pairwise_brain_mask"],
            reference_images["pairwise_brain_mask"],
        )

    # Step 4: Pairwise brain masks only (/ pairwise) --> resampled
    # Skip the transform for target modality (identity)
    if modality == target_modality or advanced_mri_skip_fail:
        transformed_masks["resampled"] = transformed_masks["pairwise_brain_mask"]
    else:
        transformed_masks["resampled"] = resample_image(
            transformed_masks["pairwise_brain_mask"],
            transforms["pairwise_brain_masks_to_resampled"],
            reference_images["resampled"],
        )

    # Step 5: Resampled --> reoriented
    transformed_masks["reoriented"] = resample_image(
        transformed_masks["resampled"],
        transforms["resampled_to_reoriented"],
        reference_images["reoriented"],
    )

    # Step 6: Reoriented --> nifti
    transformed_masks["nifti"] = apply_inverse_reorientation(
        transformed_masks["reoriented"],
        transforms["reoriented_to_nifti"],
        reference_images["reoriented"],
    )

    return transformed_masks


def load_inverse_transform(transform_file: Path):
    """Loads an ITK transform and returns its inverse.

    Args:
        transform_file (Path): Path to the transform file.

    Returns:
        itk.Transform: The inverse of the loaded ITK transform.
    """
    transform = itk.transformread(transform_file)
    inverse_transform = transform[0].GetInverseTransform()
    return inverse_transform


def load_inverse_reorientation_parameters(transform_file: Path):
    """Loads reorientation parameters from a JSON file and returns the inverse permutation.

    Args:
        transform_file (Path): Path to the JSON file containing reorientation parameters.

    Returns:
        dict: Dictionary of reorientation parameters with inverse permutation.
    """
    reorientation_parameters = IO_utils.load_json(transform_file)
    inverse_permutations = get_inverse_permutation(reorientation_parameters["permutation"])
    reorientation_parameters["permutation"] = tuple(inverse_permutations)
    return reorientation_parameters


def get_inverse_permutation(permutations: list):
    """Calculates the inverse permutation of a given list of permutations.

    Args:
        permutations (list): List of permutation indices.

    Returns:
        list: List of inverse permutation indices.
    """
    inverse_permutations = []
    permutations = np.asarray(permutations)
    for i_permutation_axis in range(len(permutations)):
        i_inverse_permutation_axis = np.argwhere(permutations == i_permutation_axis)[0][0]
        inverse_permutations.append(int(i_inverse_permutation_axis))
    return inverse_permutations


def get_inverse_patient_rotation(patient_rotations: list):
    """Computes the inverse rotation angles from the given patient rotation angles.

    Args:
        patient_rotations (list): List of patient rotation angles.

    Returns:
        list: List of inverse rotation angles.
    """
    inverse_rotations = []
    for i_rotation in patient_rotations:
        inverse_rotations.append(-1.0 * i_rotation)
    return inverse_rotations


def get_image_center(image: itk.Image):
    """Computes the center of an image based on its size, spacing, and origin.

    Args:
        image (itk.Image): ITK image object.

    Returns:
        np.ndarray: The center of the image as a numpy array.
    """
    image_size = image.GetLargestPossibleRegion().GetSize()
    image_center = image.TransformIndexToPhysicalPoint(
        [
            int(np.ceil(image_size[0] / 2)),
            int(np.ceil(image_size[1] / 2)),
            int(np.ceil(image_size[2] / 2)),
        ]
    )
    return image_center


def resample_image(mask: itk.Image, transform, reference_image: itk.Image):
    """Resamples an image to match the space of a reference image using a given transform.

    Args:
        mask (itk.Image): Image to be resampled.
        reference_image (itk.Image): Image defining the target space.
        transform (itk.Transform): Transform to be applied during resampling.

    Returns:
        itk.Image: Resampled image.
    """
    transformed_mask = itk.resample_image_filter(
        mask,
        transform=transform,
        use_reference_image=True,
        reference_image=reference_image,
        interpolator=itk.NearestNeighborInterpolateImageFunction.New(mask),
    )

    return transformed_mask


def apply_inverse_reorientation(
    mask: itk.Image, reorientation_parameters: dict, reference_image: itk.Image
) -> itk.Image:
    """Applies inverse reorientation to an image based on the given parameters.

    Args:
        mask (itk.Image): Image to be reoriented.
        reorientation_parameters (dict): Dictionary of reorientation parameters.
        reference_image (itk.Image): Image defining the target space.

    Returns:
        itk.Image: Reoriented image.
    """
    # First inverse the reorientation of the patient position

    inverse_patient_rotations = get_inverse_patient_rotation(
        reorientation_parameters["position_rotations"]
    )
    image_center = get_image_center(mask)

    # TODO
    # Here we use a double type, preferably also would
    # keep this in UC, but euler3d doesn't support that by default
    # Shouldn't be a problem but it's not very clean.
    euler_transform = itk.Euler3DTransform[itk.D].New()
    euler_transform.SetRotation(
        inverse_patient_rotations[0],
        inverse_patient_rotations[1],
        inverse_patient_rotations[2],
    )
    euler_transform.SetCenter(image_center)

    # TODO
    # The reference image we give here is the reoriented one.
    # If we give the original NIFTI one then things don't work.
    # There the flips/permutations haven't been applied.
    # So the orientation and such will not match anymore and the flips we apply
    # will not properly work.
    # We should verify whether the current solution works for
    # scans with a non-standard patient orientation.
    # Since we assign the patient orientation from the reoriented image (which is fixed)
    # by using it as a reference image, but apply rotations to the old patient orientation
    # There should be a nicer way to do this.

    mask_patient_orientation_transformed = resample_image(mask, euler_transform, reference_image)

    # Then apply inverse flips

    flip_filter = itk.FlipImageFilter[constants.ITK_UC_3D].New()
    flip_filter.SetFlipAxes(reorientation_parameters["axesflip"])
    flip_filter.SetInput(mask_patient_orientation_transformed)
    flip_filter.SetFlipAboutOrigin(False)
    flip_filter.UpdateLargestPossibleRegion()

    mask_flipped = flip_filter.GetOutput()

    # And finally reverse the permutation of axies

    permutation_filter = itk.PermuteAxesImageFilter[constants.ITK_UC_3D].New()
    permutation_filter.SetOrder(reorientation_parameters["permutation"])
    permutation_filter.SetInput(mask_flipped)
    permutation_filter.UpdateLargestPossibleRegion()

    mask_original = permutation_filter.GetOutput()
    return mask_original


if __name__ == "__main__":
    config_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    convert_nnunet_masks_all(config_path, input_folder, output_folder)