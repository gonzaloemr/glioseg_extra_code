import argparse
import os
import pathlib
import subprocess


def append_seff_stats(log_dir):
    # Iterate through each file in the directory
    for filename in os.listdir(log_dir):
        # Check if the file is a log file with the expected pattern
        if filename.endswith('.log') and '_' in filename:
            # Extract the job ID from the filename
            job_id = filename.split('_')[-1].replace('.log', '')
            
            # Get the full path to the log file
            log_file_path = os.path.join(log_dir, filename)

            # Read the content of the log file to check for existing Job Statistics
            with open(log_file_path, 'r') as log_file:
                content = log_file.read()
            
            # Check if "Job Statistics" is already in the file
            if "Job Statistics:" in content:
                print(f"Job Statistics already present in {log_file_path}, skipping...")
                continue
            
            # Run seff <job-id> to get the job statistics
            try:
                seff_output = subprocess.check_output(['seff', job_id], text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error retrieving statistics for job {job_id}: {e}")
                continue
            
            # Append the job statistics to the log file
            with open(log_file_path, 'a') as log_file:
                log_file.write('\nJob Statistics:\n')
                log_file.write(seff_output)
            
            print(f"Appended job statistics to {log_file_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Append job statistics to log files.')
    parser.add_argument('log_dir', type=str, help='Path to the directory containing log files')
    
    # Parse arguments
    args = parser.parse_args()
    log_dir = pathlib.Path(args.log_dir).resolve()

    print(log_dir)
    
    # Append job statistics to log files
    append_seff_stats(log_dir)

if __name__ == '__main__':
    main()