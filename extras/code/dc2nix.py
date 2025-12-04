import os
import subprocess


scan_types = ['T1', 'T2', 'T1GD', 'FLAIR']

# patient_dir = "/gpfs/work4/0/prjs0964/data/glioseg_data/P003012"
patient_dir = "/gpfs/home4/gmosquerarojas/glioseg_data/One_patient/Patients_dicom/EGD-0004"

if os.path.isdir(patient_dir):
    dicom_dir = os.path.join(patient_dir, 'DICOM')
    nifti_dir = os.path.join(patient_dir, 'NIFTI')

    # Ensure the NIFTI directory exists
    os.makedirs(nifti_dir, exist_ok=True)

    # Iterate over each scan type directory
    for scan_type in scan_types:
        dicom_scan_folder = os.path.join(dicom_dir, scan_type)
        if os.path.isdir(dicom_scan_folder):
            # Construct the dcm2niix command
            command = [
                'dcm2niix',
                '-9',
                '-z', 'o',
                '-f', scan_type,
                '-o', nifti_dir,
                dicom_scan_folder
            ]

            try:
                # Run the command
                subprocess.run(command, check=True)
                print(f"Successfully converted {scan_type}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {scan_type}: {e}")
