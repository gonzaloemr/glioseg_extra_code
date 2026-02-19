import shutil

from pathlib import Path


def copy_revised_patients(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_patients_dir = input_dir.joinpath("Patients")
    output_patients_dir = output_dir.joinpath("Patients")
    output_patients_dir.mkdir(parents=True,exist_ok=True)
    output_patients_gt = output_dir.joinpath("GT")

    for revised_patients in output_patients_gt.iterdir(): 
        shutil.copytree(input_patients_dir.joinpath(revised_patients.name), output_patients_dir.joinpath(revised_patients.name))
        

if __name__ == "__main__":

    input_dir = "/scratch/radv/share/glioseg/patients_clean_65_glioseg/"
    output_dir = "/scratch/radv/share/glioseg/patients_with_revised_segmentations/"

    copy_revised_patients(input_dir, output_dir)