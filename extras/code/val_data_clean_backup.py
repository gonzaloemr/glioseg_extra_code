
def validation_data_clean(original_data_dir: list[str | Path], reference_dir: str | Path, data_destination_dir: str | Path):
    """
    Cleans and organizes validation data into a structured format.

    Parameters:
    original_data_dir list[str | Path]: list with strings or Paths to the original data directories.
    reference_dir (str | Path): Path to the reference directory for data structure.
    data_destination_dir (str | Path): Path to the destination directory for cleaned data.
    """

    # data_root_dir = Path("/scratch/radv/share/glioseg/patients_65")
    # data_root_dir = Path("/data/share/IMAGO/Rotterdam_project/Gonzalo_Juancito/patients/")

    if len(original_data_dir) > 1:
        original_data_dir = [Path(data_dir) if isinstance(data_dir, str) else data_dir for data_dir in original_data_dir]
    else: 
        original_data_dir = Path(original_data_dir[0]) if isinstance(original_data_dir[0], str) else original_data_dir[0]
    
    patients_to_process = [patient_dir.name for patient_dir in reference_dir.iterdir() if patient_dir.is_dir()]

    data_clean_dir = data_destination_dir.parent.joinpath("new_run_corrected")
    patients_dir = data_clean_dir.joinpath("Patients")
    gt_data_dir = data_clean_dir.joinpath("GT")
    patients_failed_dir = data_clean_dir.joinpath("Patients_failed")
    log_files_dir = data_clean_dir.joinpath("logfiles")
    data_clean_dir.mkdir(parents=True, exist_ok=True)
    patients_dir.mkdir(parents=True, exist_ok=True)
    gt_data_dir.mkdir(parents=True, exist_ok=True)
    patients_failed_dir.mkdir(parents=True, exist_ok=True)
    log_files_dir.mkdir(parents=True, exist_ok=True)

    for patient_dir in original_data_dir: 
        for patient in patient_dir.iterdir(): 
            if patient.name in patients_to_process: 

                previous_names_files = data_clean_dir.joinpath("previous_names.txt")
                    
                patient_name = f"\nPatient {patient.name}\n"
                patient_clean_dir = patients_dir.joinpath(patient_dir.name)
                patient_clean_nifti_dir = patient_clean_dir.joinpath("NIFTI")
                patient_clean_nifti_dir.mkdir(parents=True,exist_ok=True)
                gt_clean_nifti_dir = gt_data_dir.joinpath(patient_dir.name,"NIFTI")
                gt_clean_nifti_dir.mkdir(parents=True,exist_ok=True)

                print("-"*180)
                print(patient_name)
                print("-"*180)

                with previous_names_files.open(mode="a", encoding="utf-8") as file:
                    file.write('\n'+'-'*180+'\n')
                    file.write(patient_name)
                    file.write('\n'+'-'*180+'\n')

                for sub_dir in patient_dir.iterdir():
                    
                    if sub_dir.is_dir():
                        
                        nifti_files = sorted([file.name for file in sub_dir.iterdir() if file.name.endswith(".nii.gz")])
                        if sub_dir.name == "3D_T1C" or sub_dir.name == "CET1":  
                    
                            if len(nifti_files) > 1:
                                segmentation_file = next((f for f in nifti_files if "segmentation_native_" in f), None)
                                scan_file = segmentation_file.removeprefix("segmentation_native_")
                                scan_dir_origin = sub_dir.joinpath(scan_file)
                                scan_dir_destination = patient_clean_nifti_dir.joinpath("T1GD.nii.gz")

                                gt_dir_origin = sub_dir.joinpath(segmentation_file)
                                gt_dir_destination = gt_clean_nifti_dir.joinpath("MASK.nii.gz")

                                shutil.copy(scan_dir_origin, scan_dir_destination)
                                shutil.copy(gt_dir_origin, gt_dir_destination)

                                with previous_names_files.open(mode="a", encoding="utf-8") as file:
                                    file.write(f"{scan_dir_origin} -> {scan_dir_destination}\n")
                                    file.write(f"{gt_dir_origin} -> {gt_dir_destination}\n")
                                    print(f"{scan_dir_origin} -> {scan_dir_destination}")
                                    print(f"{gt_dir_origin} -> {gt_dir_destination}")
                            else:
                                segmentation_file = next((f.name for f in list(patient_dir.iterdir()) if f.is_file() and "segmentation_native_" in f.name), None)
                                scan_file = nifti_files[0]
                                scan_dir_origin = sub_dir.joinpath(scan_file)
                                scan_dir_destination = patient_clean_nifti_dir.joinpath("T1GD.nii.gz")

                                gt_dir_origin = patient_dir.joinpath(segmentation_file)
                                gt_dir_destination = gt_clean_nifti_dir.joinpath("SEG.nii.gz")

                                shutil.copy(scan_dir_origin, scan_dir_destination)
                                shutil.copy(gt_dir_origin, gt_dir_destination)

                                with previous_names_files.open(mode="a", encoding="utf-8") as file:
                                    file.write(f"{scan_dir_origin} -> {scan_dir_destination}\n")
                                    file.write(f"{gt_dir_origin} -> {gt_dir_destination}\n")
                                    print(f"{scan_dir_origin} -> {scan_dir_destination}")
                                    print(f"{gt_dir_origin} -> {gt_dir_destination}")  
                        else: 
                            file_name = nifti_files[0]
                            scan_dir_origin = sub_dir.joinpath(file_name)
                            if "T1" in sub_dir.name:
                                scan_dir_destination = patient_clean_nifti_dir.joinpath("T1.nii.gz")
                                write_line = f"{scan_dir_origin} -> {scan_dir_destination}\n"
                                print(f"{scan_dir_origin} -> {scan_dir_destination}")
                                shutil.copy(scan_dir_origin, scan_dir_destination)
                            elif "T2" in sub_dir.name: 
                                scan_dir_destination = patient_clean_nifti_dir.joinpath("T2.nii.gz")
                                write_line = f"{scan_dir_origin} -> {scan_dir_destination}\n"
                                print(f"{file_name} -> T2.nii.gz")
                                shutil.copy(scan_dir_origin, scan_dir_destination)
                            elif "FLAIR" in sub_dir.name:
                                scan_dir_destination = patient_clean_nifti_dir.joinpath("FLAIR.nii.gz")
                                write_line = f"{scan_dir_origin} -> {scan_dir_destination}\n"
                                print(f"{scan_dir_origin} -> {scan_dir_destination}")
                                shutil.copy(scan_dir_origin, scan_dir_destination)
                            with previous_names_files.open(mode="a", encoding="utf-8") as file:
                                file.write(write_line)
