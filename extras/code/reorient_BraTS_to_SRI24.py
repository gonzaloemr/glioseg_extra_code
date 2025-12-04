from pathlib import Path
import SimpleITK as sitk
from get_information_for_reorientation import get_information_im
import json

def reorient_brats_data_to_SRI24(input_dir: str | Path, output_dir: str | Path, modality_maps: dict, sri_info: dict):
    """
    Reorient BraTS data to SRI24 space.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing BraTS patient data.
    output_dir : str | Path
        Directory to save reoriented data.
    modalit_maps : dict
        Mapping of modality names to brain mask file names. It should contain the keys from the original dataset, and the value should be the file name in the output directory.
    sri_info : dict
        Dictionary containing SRI24 image information such as Size, Origin, Spacing, Direction, Center, and Angles.
    """

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    parameter_folder = output_dir.parent / "REORIENTATION_PARAMETERS_FINAL"
    if not parameter_folder.exists():
        parameter_folder.mkdir(parents=True, exist_ok=True)
    
    for patient in input_dir.iterdir(): 

        print(f"Reorienting patient: {patient.name}", flush=True)

        patient_input_dir = input_dir / patient.name
        patient_output_dir = output_dir / patient.name
        patient_parameter_folder = parameter_folder / patient.name
        if not patient_output_dir.exists():
            patient_output_dir.mkdir(parents=True, exist_ok=True)
        if not patient_parameter_folder.exists():
            patient_parameter_folder.mkdir(parents=True, exist_ok=True)

        for file in patient_input_dir.iterdir():

            if file.name.endswith(".nii") or file.name.endswith(".nii.gz"):

                file_im = sitk.ReadImage(str(file))

                reorient_filter = sitk.DICOMOrientImageFilter()
                reorient_filter.SetDesiredCoordinateOrientation(sri_info["Orientation"])
            

                file_im_reoriented = reorient_filter.Execute(file_im)
                # file_im_reoriented.SetOrigin(sri_info["Origin"])

                sitk.WriteImage(file_im_reoriented, str(patient_output_dir / file.name))

                orientation_info = {
                    "Permutation": reorient_filter.GetPermuteOrder(),
                    "AxesFlip": reorient_filter.GetFlipAxes()
                }

                with patient_parameter_folder.joinpath(f"{file.name}_reorientation_parameters.json").open("w") as f:
                    json.dump(orientation_info, f, indent=4)


if __name__ == "__main__":


    input_dir = Path("/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/original")
    output_dir = Path("/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/reoriented_final")

    sri_24_atlas_dir = "/gpfs/work1/0/prjs0971/glioseg/data/sri24_spm8/templates"
    sri24_atlas_t1 = Path(sri_24_atlas_dir) / "T1.nii.gz"
    sri24_atlas_t2 = Path(sri_24_atlas_dir) / "T2.nii.gz"
    sri24_brain_mask_t1 = Path(sri_24_atlas_dir) / "combined_brain_mask.nii.gz"

    info_for_reorientation = get_information_im(sri24_atlas_t1, sri24_brain_mask_t1)

    SRI24_SIZE = info_for_reorientation["Size"]
    SRI24_ORIGIN = info_for_reorientation["Origin"]
    SRI24_ORIENTATION = info_for_reorientation["Orientation"]
    SRI24_SPACING = info_for_reorientation["Spacing"]
    SRI24_DIRECTION = info_for_reorientation["Direction"]
    SRI24_CENTER = info_for_reorientation["Center"]
    SRI24_ANGLES = info_for_reorientation["Angles"]

    BRATS_SIZE = (240, 240, 155)
    BRATS_ORIGIN = (-0.0, -239.0, 0.0)
    BRATS_SPACING = (1.0, 1.0, 1.0)
    BRATS_DIRECTION = (1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0)
    BRATS_ORIENTATION = "LPS"
    BRATS_ANGLES = [-0.0, 0.0, 0.0]

    modality_maps = {
        "t1n": "T1",
        "t1c": "T1GD",
        "t2w": "T2",
        "t2f": "FLAIR"
    }

    reorient_brats_data_to_SRI24(input_dir, output_dir, modality_maps, info_for_reorientation)

    


