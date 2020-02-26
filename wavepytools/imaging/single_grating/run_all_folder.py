import os, time, shutil
from SingleGrating_folder import SingleGrating
from fileread_postprocess_func import line_profile_process

escluded_directories = ['completed', '.DS_Store', 'failed']

def process_scan_files(scan_files):
    # print("launch wavepy on: ", scan_files)

    # start to process the wavefront data
    SingleGrating(scan_files)

    # after the wavefront is processed, collect the data to a single file
    # get the folder path to the images.
    file_directory = os.path.dirname(scan_files[0])
    line_profile_process(file_directory)
    # time.sleep(10)


def process_scan_directory(scan_directory, completed_scan_directory, failed_scan_directory):
    scan_files = [os.path.join(scan_directory, scan_file) for scan_file in os.listdir(scan_directory) if
                  os.path.isfile(os.path.join(scan_directory, scan_file)) and
                  os.path.splitext(scan_file)[1]==".tif"]

    if len(scan_files) > 0:

            try:
                process_scan_files(scan_files)

                print("Wavepy Completed")

                shutil.move(scan_directory, completed_scan_directory)
            except Exception as exception:
                print("Wavepy Failed", str(exception))

                shutil.move(scan_directory, failed_scan_directory)


def look_for_scan_dir(scan_home_directory, completed_scan_directory, failed_scan_directory):
    scan_directories = [os.path.join(scan_home_directory, scan_directory) for scan_directory in os.listdir(scan_home_directory) if
                        os.path.isdir(os.path.join(scan_home_directory, scan_directory))
                        and scan_directory not in escluded_directories]

    # we take only the first found, then repeat the monitor
    if len(scan_directories) > 0:
        for scan_directory in scan_directories:
            # current_scan_directory = sorted(scan_directories)[0]
            print(scan_directory)

            process_scan_directory(scan_directory, completed_scan_directory, failed_scan_directory)


import sys

if __name__=="__main__":
    scan_home_directory = sys.argv[1]

    completed_scan_directory = os.path.join(scan_home_directory, "completed")
    failed_scan_directory = os.path.join(scan_home_directory, "failed")

    if not os.path.exists(completed_scan_directory): os.mkdir(completed_scan_directory)
    if not os.path.exists(failed_scan_directory): os.mkdir(failed_scan_directory)
    look_for_scan_dir(scan_home_directory, completed_scan_directory, failed_scan_directory)
