from pathlib import Path

import SimpleITK as sitk


data_dir = Path("/data/share/IMAGO/Rotterdam_project/patients/IM0009/")


CE_T1_or = sitk.ReadImage(data_dir.joinpath("3D_T1C","T13Ddis_901.nii.gz"))
T1_or = sitk.ReadImage(data_dir.joinpath("2D_T1","T1W_SE_TRA_interleaved_501.nii.gz"))
T2_or = sitk.ReadImage(data_dir.joinpath("T2", "T2W_TSE_TRA_+C_701.nii.gz"))
FLAIR_or = sitk.ReadImage(data_dir.joinpath("3DFLAIR", "3d_flair_SAG_401.nii.gz"))

CE_T1_reg = sitk.ReadImage(data_dir.joinpath("CET1.nii.gz"))
T1_reg = sitk.ReadImage(data_dir.joinpath("T2_2_CET1.nii.gz"))
T2_reg = sitk.ReadImage(data_dir.joinpath("T2_2_CET1.nii.gz"))
FLAIR_reg = sitk.ReadImage(data_dir.joinpath("FLAIR_2_CET1.nii.gz"))

print(f"CE_T1_or origin: {CE_T1_or.GetOrigin()}")
print(f"CE_T1_spacing: {CE_T1_or.GetSpacing()}")
print(f"CE_T1_direction: {CE_T1_or.GetDirection()}")

print(f"T1_or origin: {T1_or.GetOrigin()}")
print(f"T1_spacing: {T1_or.GetSpacing()}")
print(f"T1_direction: {T1_or.GetDirection()}")

print(f"T2_or origin: {T2_or.GetOrigin()}")
print(f"T2_spacing: {T2_or.GetSpacing()}")
print(f"T2_direction: {T2_or.GetDirection()}") 

print(f"FLAIR_or origin: {FLAIR_or.GetOrigin()}")
print(f"FLAIR_spacing: {FLAIR_or.GetSpacing()}")
print(f"FLAIR_direction: {FLAIR_or.GetDirection()}")    

print(f"CE_T1_reg origin: {CE_T1_reg.GetOrigin()}")
print(f"CE_T1_reg spacing: {CE_T1_reg.GetSpacing()}")
print(f"CE_T1_reg direction: {CE_T1_reg.GetDirection()}")

print(f"T1_reg origin: {T1_reg.GetOrigin()}")
print(f"T1_reg spacing: {T1_reg.GetSpacing()}")
print(f"T1_reg direction: {T1_reg.GetDirection()}")

print(f"T2_reg origin: {T2_reg.GetOrigin()}")
print(f"T2_reg spacing: {T2_reg.GetSpacing()}")
print(f"T2_reg direction: {T2_reg.GetDirection()}")

print(f"FLAIR_reg origin: {FLAIR_reg.GetOrigin()}")
print(f"FLAIR_reg spacing: {FLAIR_reg.GetSpacing()}")
print(f"FLAIR_reg direction: {FLAIR_reg.GetDirection()}")

