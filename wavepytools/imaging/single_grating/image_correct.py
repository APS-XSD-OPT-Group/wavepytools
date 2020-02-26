'''
this file provide a function which can do pre-processing for images.
    1. dark image correction
    2. flat image correction
    3. distortion correction
    for terminal, here is how to use it:
        image_correct image_path, dark_image_path, flat_image_path, distort_data_path, output_path
        argv[0]:    raw image path
        argv[1]:    dark image path
        argv[2]:    flat image path
        argv[3]:    distortion map data path
        argv[4]:    path to save processed images
        argv[5]:    


        if the dark, flat or distortion map is None, means to do nothing with this correction
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import imageio
import glob
import tkinter as tk
from tkinter import filedialog
# import the image correction functions from the distort_correct.py file
from distort_correct import dark_err, flat_err, image_corr, distort_corr


def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)


def gui_load_data_finename(directory='', title="File name with Data"):
    
    originalDir = os.getcwd()

    if directory != '':

        if os.path.isdir(directory):
            os.chdir(directory)
        else:
            prColor("WARNING: Directory " + directory + " doesn't exist.", 'red')
            prColor("MESSAGE: Using current working directory " + originalDir, 'yellow')

    root = tk.Tk(title)
    root.withdraw()
    fname1 = filedialog.askopenfilename()
    # fname1 = easyqt.get_file_names(title)

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1[0]

    os.chdir(originalDir)

    return fname_last

def gui_load_data_directory(directory='', title="File name with Data"):
    
    originalDir = os.getcwd()

    if directory != '':

        if os.path.isdir(directory):
            os.chdir(directory)
        else:
            prColor("WARNING: Directory " + directory + " doesn't exist.", 'red')
            prColor("MESSAGE: Using current working directory " + originalDir, 'yellow')
    
    root = tk.Tk(title)
    root.withdraw()
    fname1 = filedialog.askdirectory()
    # fname1 = easyqt.get_directory_name(title)

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1

    os.chdir(originalDir)

    return fname_last


def img_corret(fname_dark, fname_flat, fname_distort, fname_img, fname_output, position):
    '''
        here is the code to correct the raw images and also use dark and flat image to correct the background and noise
        input:
            fname_dark:             the path to dark image
            fname_flat:             the path to flat image, a folder or a image
            fname_distort:          the path to the distort data, should be a folder
            fname_img:              the path to the raw images
            fname_output:           the path to the saving result folder
            position:               the corresponding position of the distort data in the raw image
                                    if should be same the the DetectorDistortion.py parameters

    '''
    if fname_dark is None:
        dark_data = None
    else:
        dark_data = dark_err(fname_dark)
    if fname_flat is None:
        flat_data = None
    else:
        flat_data = flat_err(fname_flat)

    # load data for the raw image or images
    if os.path.isdir(fname_img):
        # if the path to raw image is a folder
        listOfFiles = glob.glob(fname_img + '/*.tif')
        listOfFiles.sort()
        listOfData = []
        filename_origin = []
        for fname in listOfFiles:
            prColor('MESSAGE: Open File ' + fname, 'green')
            temp_data = tiff.imread(fname)
            listOfData.append(temp_data)
            filename_origin.append(os.path.split(fname)[-1])

        img_data = np.array(listOfData)

    elif os.path.isfile(fname_img):
        # if the path to raw image is a file directory
        img_data = tiff.imread(fname_img)
        filename_origin = os.path.split(fname_img)[-1]
    elif isinstance(fname_img, np.ndarray):
        img_data = np.copy(fname_img)
        filename_origin = []
    else:
        prColor('Error: wrong image path', 'red')
        sys.exit()

    img_corr = image_corr(img_data, dark_data, flat_data)

    if fname_distort == None:
        img_last = np.copy(img_corr)
    else:
        if os.path.isdir(fname_distort):
            # if the path to distortion image is a folder
            listOfDist = glob.glob(fname_distort + '/*.dat')
            if len(listOfDist) == 0:
                listOfDist = glob.glob(fname_distort + '/*.txt')
            listOfDist.sort()
            if len(listOfDist) > 2:
                prColor('Error: too many files in the distortion data folder', 'red')
            elif len(listOfDist) == 0:
                prColor('Error: no such directory or files for distortion data', 'red')
        else:
            prColor('Error: Wrong path to the distortion data folder', 'red')
        print(listOfDist)
        img_last = distort_corr(img_corr, listOfDist, position)
    
    img_last = np.array(img_last, dtype='uint16')

    if os.path.isdir(fname_output):
        subdir = '/corrected_images/'
        if not os.path.exists(fname_output+subdir):
            os.makedirs(fname_output+subdir, exist_ok=True)
    
        if len(filename_origin) == 0:
            file_name = 'correct_image'
        else:
            file_name = filename_origin
        if len(img_last.shape) == 2:
            tiff.imsave(fname_output+subdir+file_name, img_last)
        elif len(img_last.shape) == 3:
            for kk in range(img_last.shape[0]):
                plt.figure()
                plt.imshow(img_last[kk])
                if len(filename_origin)==0:
                    name_last = '_'+str(kk)
                    tiff.imsave(fname_output+subdir+file_name+name_last+'.tif', img_last[kk])
                    prColor('Message: save image ' + fname_output+subdir+file_name+name_last+'.tif', 'green')
                else:
                    tiff.imsave(fname_output+subdir+file_name[kk], img_last[kk])
                    prColor('Message: save image ' + fname_output+subdir+file_name[kk], 'green')
                
        else:
            prColor('Error: wrong processed data', 'red')
            sys.exit()

if __name__ == "__main__":
    if len(sys.argv) == 6:
        # for terminal mode
        fname_img, fname_dark, fname_flat, fname_distort, fname_output = sys.argv[1:6]
        print(fname_img, '\n',fname_dark, '\n',fname_flat, '\n',fname_distort, '\n',fname_output)
        
        # here is the distortion data position in the raw image. [Y_start, Y_end, X_start, X_end]
        position = [50, 2500, 50, 2100]
        img_corret(fname_dark, fname_flat,fname_distort, fname_img, fname_output, position)

    elif len(sys.argv) == 1:
        # for gui mode
        # load the raw image
        print('\033[32m'+'MESSAGE: select dark image'+'\033[0m')
        fname_dark = gui_load_data_finename('', 'load dark image')
        print('\033[32m'+'MESSAGE: select flat image' +'\033[0m')
        fname_flat = gui_load_data_finename('', 'load flat image')
        print('\033[32m'+'MESSAGE: directory to distortion map'+'\033[0m')
        fname_distort = gui_load_data_directory('', 'directory to distortion map')
        print('\033[32m'+'MESSAGE: directory to raw image'+'\033[0m')
        fname_img = gui_load_data_directory('', 'directory to raw image')
        print('\033[32m'+'MESSAGE: directory to result folder'+'\033[0m')
        fname_output = gui_load_data_directory('', 'directory to save processed images')

        # here is the distortion data position in the raw image. [Y_start, Y_end, X_start, X_end]
        position = [50, 2500, 50, 2100]
        img_corret(fname_dark, fname_flat,fname_distort, fname_img, fname_output, position)
        
        
    
    

