import os
import pathlib
import re
import sys

from datetime import timedelta


def parse_log_file(filepath):
    memory_utilized = None
    wall_clock_time = None
    
    with open(filepath, 'r') as file:
        for line in file:
            if "Memory Utilized" in line:
                gb_match = re.search(r"Memory Utilized:\s*([\d.]+)\s*GB", line)
                mb_match = re.search(r"Memory Utilized:\s*([\d.]+)\s*MB", line)
                if gb_match:
                    memory_utilized = float(gb_match.group(1))
                elif mb_match:
                    memory_utilized = float(mb_match.group(1)) / 1024  # Convert MB to GB
            elif "Job Wall-clock time" in line:
                match = re.search(r"Job Wall-clock time:\s*(\d{2}:\d{2}:\d{2})", line)
                if match:
                    wall_clock_time = match.group(1)
    
    return memory_utilized, wall_clock_time

def convert_to_timedelta(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def main(log_dir):
    log_dir = pathlib.Path(log_dir)
    output_filepath = log_dir / "job_statistics.txt"

    total_memory = 0.0
    total_wall_clock_time = timedelta()

    with open(output_filepath, 'w') as output_file:
        output_file.write("Filename\tMemory Utilized (GB)\tWall Clock Time\n")
        
        for log_file in log_dir.glob("*.log"):
            memory_utilized, wall_clock_time = parse_log_file(log_file)
            
            if memory_utilized is not None and wall_clock_time:
                total_memory += memory_utilized
                total_wall_clock_time += convert_to_timedelta(wall_clock_time)
                output_file.write(f"{log_file.name}\t{memory_utilized:.2f}\t{wall_clock_time}\n")
            else:
                output_file.write(f"{log_file.name}\tN/A\tN/A\n")
                if memory_utilized is None:
                    print(f"Debug: 'Memory Utilized' not found or unrecognized in {log_file}")
                if not wall_clock_time:
                    print(f"Debug: 'Job Wall-clock time' not found in {log_file}")
        
        # Write totals
        total_wall_clock_str = str(total_wall_clock_time)
        output_file.write(f"Total\t{total_memory:.2f}\t{total_wall_clock_str}\n")
    
    print(f"Job statistics written to {output_filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python job_stats.py <path_to_log_directory>")
        sys.exit(1)
    
    log_directory = sys.argv[1]
    main(log_directory)