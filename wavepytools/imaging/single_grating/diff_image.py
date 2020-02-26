'''
    this function provides a tool to get differential image. the image type is tiff
    this function can be used as gui mode or terminal mode
    for terminal mode, here is how to use it:
        image_crop path_to_image_folder path_to_output_folder crop_size_col crop_size_row col_shift row_shift

        description:
        sample:     raw image folder with sample
        nosample:     raw image without sample
        dark:     raw dark image
        path_output:    output image folder

'''

import os
import sys
import skimage
import numpy as np
import dxchange
import tifffile as tiff
from wavepy.utils import easyqt
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import glob

def gui_load_data_directory(directory='', title="File name with Data"):
    
    originalDir = os.getcwd()

    fname1 = easyqt.get_directory_name(title)

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1

    os.chdir(originalDir)

    return fname_last

def gui_load_data_finename(directory='', title="File name with Data"):
    
    originalDir = os.getcwd()

    if directory != '':

        if os.path.isdir(directory):
            os.chdir(directory)
        else:
            wpu.print_red("WARNING: Directory " + directory + " doesn't exist.")
            wpu.print_blue("MESSAGE: Using current working directory " +
                           originalDir)

    fname1 = easyqt.get_file_names(title)

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1[0]

    os.chdir(originalDir)

    return fname_last


if __name__ == "__main__":
    '''
        input:
        sample:     raw image folder with sample
        nosample:     raw image without sample
        dark:     raw dark image
        path_output:    output image folder
    '''
    if len(sys.argv) == 5:
        '''
            if the passing parameter has five parameters, means it's under terminal mode
        '''
        sample_path, nosample_path, dark_path, path_output = sys.argv[1:5]

        # read sample image
        sample_img = tiff.imread(sample_path)
        sample_img = np.array(sample_img)
        # read no sample image
        nosample_img = tiff.imread(nosample_path)
        nosample_img = np.array(nosample_img)
        # read dark image
        dark_img = tiff.imread(dark_path)
        dark_img = np.array(dark_img)

        # process the data
        # sample_img = sample_img - dark_img
        # nosample_img = nosample_img - dark_img
        sample_img = np.abs((sample_img - np.amin(sample_img))/(np.amax(sample_img) - np.amin(sample_img)))
        nosample_img = np.abs((nosample_img - np.amin(nosample_img))/(np.amax(nosample_img) - np.amin(nosample_img)))
        color_type = 'gist_heat'
        # plt.subplot(121)
        # plt.imshow(sample_img, cmap=plt.get_cmap(color_type), vmax=abs(sample_img).max(), vmin=-abs(sample_img).max())
        # plt.subplot(122)
        # plt.imshow(nosample_img, cmap=plt.get_cmap(color_type), vmax=abs(nosample_img).max(), vmin=-abs(nosample_img).max())

        # plt.show()
        result = np.abs((sample_img-dark_img)/(nosample_img-dark_img))
        result = np.abs((result - np.amin(result))/(np.amax(result) - np.amin(result))) * 40000
        result_log = np.log10(result+1e-20)
        img_result = np.array(result, dtype='uint16')
        img_result_log = np.array(result_log, dtype='uint16')

        plt.imshow(result, cmap=plt.get_cmap(color_type), vmax=abs(result).max(), vmin=-abs(result).max())
        plt.title('Siemens star image (linear)')
        # plt.show()
        # plt.imshow(result_log, cmap=plt.get_cmap(color_type))
        # plt.title('Siemens star image (log10)')
        # plt.show()

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        
        # file_name = path_output + '/result_sample_image_640mm'
        # file_number = 640
        # while os.path.isfile(file_name+'.tif'):
        #     file_number = file_number + 10
        #     file_name = path_output + '/result_sample_image_' + str(file_number)+'mm'
            
        file_name = path_output + '/result_sample_image'
        tiff.imsave(file_name + '.tif', img_result)
        plt.imsave(file_name + '.png', result, cmap=plt.get_cmap(color_type), format='png')
        # plt.imsave(path_output + '/result_sample_image_log10.png', result_log, cmap=plt.get_cmap(color_type), format='png')
        tiff.imsave(file_name + '_log10' + '.tif',img_result)
        print('\033[32m' + 'MESSAGE: File ' + file_name + '.tif'
                '   saved' + '\033[0m')


    elif len(sys.argv) == 1:
        '''
            if only one parameter, it's gui mode to get the folder path
        '''

        print('\033[32m' + 'MESSAGE: select raw image with sample' + '\033[0m')
        sample_path = gui_load_data_finename('', 'load sample image')

        print('\033[32m' + 'MESSAGE: select raw image without sample' + '\033[0m')
        nosample_path = gui_load_data_finename('', 'load no sample image')

        print('\033[32m' + 'MESSAGE: select raw dark image' + '\033[0m')
        dark_path = gui_load_data_finename('', 'load dark image')

        print('\033[32m' + 'MESSAGE: select output folder' + '\033[0m')
        path_output = gui_load_data_directory('', 'select output folder')

        # read sample image
        sample_img = tiff.imread(sample_path)
        sample_img = np.array(sample_img)
        # read no sample image
        nosample_img = tiff.imread(nosample_path)
        nosample_img = np.array(nosample_img)
        # read dark image
        dark_img = tiff.imread(dark_path)
        dark_img = np.array(dark_img)

        # process the data
        # sample_img = sample_img - dark_img
        # nosample_img = nosample_img - dark_img
        sample_img = np.abs((sample_img - np.amin(sample_img))/(np.amax(sample_img) - np.amin(sample_img)))
        nosample_img = np.abs((nosample_img - np.amin(nosample_img))/(np.amax(nosample_img) - np.amin(nosample_img)))
        color_type = 'gist_heat'
        plt.subplot(121)
        plt.imshow(sample_img, cmap=plt.get_cmap(color_type), vmax=abs(sample_img).max(), vmin=-abs(sample_img).max())
        plt.subplot(122)
        plt.imshow(nosample_img, cmap=plt.get_cmap(color_type), vmax=abs(nosample_img).max(), vmin=-abs(nosample_img).max())

        plt.show()
        result = np.abs((sample_img-dark_img)/(nosample_img-dark_img))
        result = np.abs((result - np.amin(result))/(np.amax(result) - np.amin(result))) * 40000
        result_log = np.log10(result+1e-13)
        img_result = np.array(result, dtype='uint16')
        img_result_log = np.array(result_log, dtype='uint16')

        plt.imshow(result, cmap=plt.get_cmap(color_type), vmax=abs(result).max(), vmin=-abs(result).max())
        plt.title('Siemens star image (linear)')
        plt.show()
        plt.imshow(result_log, cmap=plt.get_cmap('hot'), vmax=np.amax(result_log), vmin=np.amin(result_log))
        plt.title('Siemens star image (log10)')
        plt.show()

        tiff.imsave(
                path_output + '/result_sample_image' + '.tif',
                img_result)
        plt.imsave(path_output + '/result_sample_image.png', result, cmap=plt.get_cmap(color_type), format='png')
        plt.imsave(path_output + '/result_sample_image_log10.png', result_log, cmap=plt.get_cmap(color_type), format='png')
        tiff.imsave(
                path_output + '/result_sample_image_log10' + '.tif',
                img_result)
        print('\033[32m' + 'MESSAGE: File ' + path_output + '/result_sample_image' + '.tif'
                '   saved' + '\033[0m')
        
