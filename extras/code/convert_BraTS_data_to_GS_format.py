from pathlib import Path 
import os 
import SimpleITK as sitk
import numpy as np
import shutil


def convert_BraTS_data_to_GS_format(input_dir: str | Path, output_dir: str | Path, modality_maps: dict):

    if not isinstance(input_dir, Path):
        input_dir = Path(input_dir)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_dir = output_dir.parent / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for patient in input_dir.iterdir():
        
        patient_name = patient.name
        patient_output_dir = output_dir / patient_name / "NIFTI"
        gt_patient_dir = gt_dir / patient_name / "NIFTI"
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        gt_patient_dir.mkdir(parents=True, exist_ok=True)

        for modality_origin, modality_dest in modality_maps.items(): 

            modality_origin_file = patient / f"{patient_name}-{modality_origin}.nii.gz"
            modality_dest_file = patient_output_dir / f"{modality_dest}.nii.gz"

            if not modality_origin_file.exists():
                print(f"Warning: {modality_origin_file} does not exist. Skipping.")
            else: 
                shutil.copy(modality_origin_file, modality_dest_file)
                print(f"Copied {modality_origin_file} to {modality_dest_file}")
            
            gt_origin_file = patient / f"{patient_name}-seg_nec_relabel.nii.gz"
            gt_dest_file = gt_patient_dir /  f"MASK.nii.gz"
            
            if not gt_origin_file.exists():
                print(f"Warning: {gt_origin_file} does not exist. Skipping.")
            else:
                shutil.copy(gt_origin_file, gt_dest_file)
                print(f"Copied {gt_origin_file} to {gt_dest_file}")


if __name__ == "__main__":
    
    
    input_dir = "/gpfs/work1/0/prjs0971/glioseg/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/original"
    output_dir = "/gpfs/work1/0/prjs0971/glioseg/data/BraTS2023_relabeled_2/Patients"

    modality_maps = {
        "t1n": "T1",
        "t1c": "T1GD",
        "t2w": "T2",
        "t2f": "FLAIR"
    }
    convert_BraTS_data_to_GS_format(input_dir, output_dir, modality_maps)