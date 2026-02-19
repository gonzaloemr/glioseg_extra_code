import shutil

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


data_root_dir = Path("/scratch/radv/share/glioseg/patients_65")
# data_root_dir = Path("/data/share/IMAGO/Rotterdam_project/Gonzalo_Juancito/patients/")
# scratch_dir = Path("/scratch/radv/gemosquerarojas/60_patients")
# shutil.move(scratch_dir,glioseg_dir)
# shutil.copytree(data_root_dir,scratch_dir)
data_clean_dir = data_root_dir.parent.joinpath("Patients_65_clean")
patients_dir = data_clean_dir.joinpath("Patients")
gt_data_dir = data_clean_dir.joinpath("GT")
patients_failed_dir = data_clean_dir.joinpath("Patients_failed")
log_files_dir = data_clean_dir.joinpath("logfiles")
data_clean_dir.mkdir(parents=True, exist_ok=True)
patients_dir.mkdir(parents=True, exist_ok=True)
gt_data_dir.mkdir(parents=True, exist_ok=True)
patients_failed_dir.mkdir(parents=True, exist_ok=True)
log_files_dir.mkdir(parents=True, exist_ok=True)

previous_names_files = data_clean_dir.joinpath("previous_names.txt")

# for patient_dir in data_root_dir.iterdir():
#     patient_name = f"\nPatient {patient_dir.name}\n"
#     print(patient_name)
#     with previous_names_files.open(mode="a", encoding="utf-8") as file:
#         file.write('\n'+'-'*50+'\n')
#         file.write(patient_name)
#         file.write('\n'+'-'*50+'\n')
        
#     patient_dir_clean = patients_dir.joinpath(patient_dir.name)
#     patient_dir_clean.mkdir(parents=True, exist_ok=True)
#     patient_dir_clean_nifti = patient_dir_clean.joinpath("NIFTI")
#     patient_dir_clean_nifti.mkdir(parents=True, exist_ok=True)
#     for patient_sub_dirs in patient_dir.iterdir():
#         if patient_sub_dirs.is_dir():
#             for file in patient_sub_dirs.iterdir():
#                 if file.name.endswith("nii.gz"):

for patient_dir in data_root_dir.iterdir():
    patient_name = f"\nPatient {patient_dir.name}\n"
    print(patient_name)
    with previous_names_files.open(mode="a", encoding="utf-8") as file:
        file.write('\n'+'-'*50+'\n')
        file.write(patient_name)
        file.write('\n'+'-'*50+'\n')
        
    patient_dir_clean = patients_dir.joinpath(patient_dir.name)
    patient_dir_clean.mkdir(parents=True, exist_ok=True)
    patient_dir_clean_nifti = patient_dir_clean.joinpath("NIFTI")
    patient_dir_clean_nifti.mkdir(parents=True, exist_ok=True)
    for patient_sub_dirs in patient_dir.iterdir():
        file_name = patient_sub_dirs.name
        with previous_names_files.open(mode="a", encoding="utf-8") as file:
            if file_name.endswith(".nii.gz") and "skullstripped" not in file_name:
                if file_name.split("_")[0] == "T1":
                    file_destination = patient_dir_clean_nifti.joinpath("T1.nii.gz")
                    file.write(f"{file_name} -> T1.nii.gz\n")
                elif file_name.split("_")[0] == "T2":
                    file_destination = patient_dir_clean_nifti.joinpath("T2.nii.gz")
                    file.write(f"{file_name} -> T2.nii.gz\n")
                elif file_name.split("_")[0] == "FLAIR":
                    file_destination = patient_dir_clean_nifti.joinpath("FLAIR.nii.gz")
                    file.write(f"{file_name} -> FLAIR.nii.gz\n")
                elif file_name.split("_")[0] == "segmentation":
                    gt_patient_dir = gt_data_dir.joinpath(patient_dir.name)
                    gt_patient_dir.mkdir(parents=True, exist_ok=True)
                    gt_patient_dir_nifti = gt_patient_dir.joinpath("NIFTI")
                    gt_patient_dir_nifti.mkdir(parents=True, exist_ok=True)
                    file_destination = gt_patient_dir_nifti.joinpath("MASK.nii.gz")
                    file.write(f"{file_name} -> MASK.nii.gz\n")
                else:
                    file_destination = patient_dir_clean_nifti.joinpath("T1GD.nii.gz")
                    file.write(f"{file_name} -> T1GD.nii.gz\n")
                shutil.copy(patient_sub_dirs,file_destination)
                print(file_name)
    with previous_names_files.open(mode="a", encoding="utf-8") as file:
        file.write('\n'+'-'*50+'\n')
    print('-'*50)

# im_path = "/data/share/IMAGO/Rotterdam_project/Gonzalo_Juancito/patients/IM0009/3DFLAIR/3d_flair_SAG_401.nii.gz"
# im_path_2 = "/data/share/IMAGO/Rotterdam_project/Gonzalo_Juancito/patients/IM0009/2D_T1/T1W_SE_TRA_interleaved_501.nii.gz"
# image = sitk.ReadImage(im_path_2)

# img_arr = sitk.GetArrayFromImage(image)
# # img_arr = np.transpose(img_arr, (2,1,0))
# print(f"Image shape {img_arr.shape}")
# plt.figure()
# plt.imshow((np.flipud(img_arr[14,:,:])),cmap="gray")
# plt.savefig("slice.png")
# plt.close()