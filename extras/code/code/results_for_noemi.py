import argparse
from pathlib import Path
import shutil 

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i_d",
    "--input_dir",
    required=True,
    help="input directory containing patient folders",
    metavar="input directory",
    dest="input_dir",
    type=str,
)


input_dir = Path(arg_parser.parse_args().input_dir)
output_dir = input_dir.parent / "SEGMENTATIONS" 

output_dir.mkdir(exist_ok=True, parents=True)


def generate_results_for_noemi():

    for patient in Path(input_dir).iterdir():
        if patient.is_dir():
            print(f"Processing patient: {patient.name}")
            patient_output_dir = output_dir / patient.name
            patient_output_dir.mkdir(exist_ok=True, parents=True)
            t1_segmentation_file = patient.joinpath("SEGMENTATIONS", "NIFTI", "mask_tumor_ensemble_relabelled_T1.nii.gz")
            t1gd_segmentation_file = patient.joinpath("SEGMENTATIONS", "NIFTI", "mask_tumor_ensemble_relabelled_T1GD.nii.gz")
            t2_segmentation_file = patient.joinpath("SEGMENTATIONS", "NIFTI", "mask_tumor_ensemble_relabelled_T2.nii.gz")
            flair_segmentation_file = patient.joinpath("SEGMENTATIONS", "NIFTI", "mask_tumor_ensemble_relabelled_FLAIR.nii.gz")
            shutil.copy(t1_segmentation_file, patient_output_dir.joinpath("mask_tumor_ensemble_relabelled_T1.nii.gz"))
            shutil.copy(t1gd_segmentation_file, patient_output_dir.joinpath("mask_tumor_ensemble_relabelled_T1GD.nii.gz"))
            shutil.copy(t2_segmentation_file, patient_output_dir.joinpath("mask_tumor_ensemble_relabelled_T2.nii.gz"))
            shutil.copy(flair_segmentation_file, patient_output_dir.joinpath("mask_tumor_ensemble_relabelled_FLAIR.nii.gz"))
            print(f"Copied segmentations to {patient_output_dir}")

    # Now zip the output directory
    shutil.make_archive(str(output_dir), 'zip', str(output_dir))
            
def main():
    generate_results_for_noemi()

if __name__ == "__main__":

    main()