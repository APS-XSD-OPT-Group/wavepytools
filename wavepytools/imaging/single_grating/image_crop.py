'''
    this function provides a tool to crop images. the image type is tiff
    this function can be used as gui mode or terminal mode
    for terminal mode, here is how to use it:
        image_crop path_to_image_folder path_to_output_folder crop_size_col crop_size_row col_shift row_shift

        description:
        path_image:     raw image folder
        path_output:    output image folder
        col_size:       col size of the image
        row_size:       row size of the image
        col_shift:      center shift along col side
        row_shift:      center shift along row side
'''

import os
import sys
import scipy.ndimage as sn
import numpy as np
# import dxchange
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def gui_load_data_directory(directory='', title="File name with Data"):

    originalDir = os.getcwd()

    root = tk.Tk(title)
    root.withdraw()
    fname1 = filedialog.askdirectory()

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1

    os.chdir(originalDir)

    return fname_last


if __name__ == "__main__":
    '''
        input:
        path_img:       image folder
        path_output:    output image folder
        col_size:       col size of the image
        row_size:       row size of the image
        col_shift:      center shift along col side
        row_shift:       center shift along row side
        method_c:       the method to find the center of the image
                        'mess': use the mess center as the center
                        'image_center': use the image center as the center
                        'peak': use the peak postion as the center
    '''
    # sort the file with modification time
    time_sort = True
    if len(sys.argv) == 8:
        '''
            if the passing parameter has six parameters, means it's under terminal mode
        '''
        path_img, path_output = sys.argv[1:3]

        col_size = int(sys.argv[3])
        row_size = int(sys.argv[4])
        col_shift = int(sys.argv[5])
        row_shift = int(sys.argv[6])
        method_c = sys.argv[7]
        # read all the images under that folder
        listOfFiles = glob.glob(path_img + '/*.tif')
        if time_sort:
            listOfFiles.sort(key=os.path.getmtime)
        else:
            listOfFiles.sort()
        listOfData = []
        filename_origin = []
        for fname in listOfFiles:
            print('\033[32m' + 'MESSAGE: Open File ' + fname + '\033[0m')
            # temp_data = dxchange.read_tiff(fname)
            temp_data = tiff.imread(fname)
            listOfData.append(temp_data)
            filename_origin.append(os.path.split(fname)[-1])

        img_data = np.array(listOfData)

        # find the center and crop it to new data
        col_pos = np.arange(0, col_size) - round(col_size / 2) + col_shift
        row_pos = np.arange(0, row_size) - round(row_size / 2) + row_shift
        img_crop = []
        for s_image in img_data:
            s_image = np.pad(s_image, (row_size, col_size), mode='edge')
            temp = s_image - np.amax(s_image) / 5
            if method_c == 'mess':
                M = sn.measurements.center_of_mass(temp)
                centroid_row = int(round(M[0]))
                centroid_col = int(round(M[1]))
            elif method_c == 'peak':
                M = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                centroid_row = M[0]
                centroid_col = M[1]
            elif method_c == 'image_center':
                centroid_row = int(round(temp.shape[0]/2))
                centroid_col = int(round(temp.shape[1]/2))
            else:
                print('wrong crop method')
                sys.exit()
            img_crop.append(
                s_image[centroid_row + row_pos[0]:centroid_row + row_pos[-1]+1,
                        centroid_col + col_pos[0]:centroid_col + col_pos[-1]+1])
        img_crop = np.array(img_crop, dtype='uint16')
        sub_folder = '/cropped/'
        if not os.path.exists(path_output + sub_folder):
            os.makedirs(path_output + sub_folder, exist_ok=True)
        for kk, s_image in enumerate(img_crop):

            tiff.imsave(
                path_output + sub_folder + filename_origin[kk],
                s_image)
            print('\033[32m' + 'MESSAGE: File ' + path_output + sub_folder +
                  filename_origin[kk] + 
                  '   saved' + '\033[0m')

    elif len(sys.argv) == 6:
        '''
            if only five parameter, it's gui mode to get the folder path
        '''
        col_size = int(sys.argv[1])
        row_size = int(sys.argv[2])
        col_shift = int(sys.argv[3])
        row_shift = int(sys.argv[4])
        method_c = sys.argv[5]

        print('\033[32m' + 'MESSAGE: select raw image folder' + '\033[0m')
        path_img = gui_load_data_directory('', 'load raw image folder')
        print('\033[32m' + 'MESSAGE: select output croped image folder' +
              '\033[0m')
        path_output = gui_load_data_directory('', 'output croped image folder')

        # read all the images under that folder
        listOfFiles = glob.glob(path_img + '/*.tif')
        if time_sort:
            listOfFiles.sort(key=os.path.getmtime)
        else:
            listOfFiles.sort()
        listOfData = []
        filename_origin = []
        for fname in listOfFiles:
            print('\033[32m' + 'MESSAGE: Open File ' + fname + '\033[0m')
            # temp_data = dxchange.read_tiff(fname)
            temp_data = tiff.imread(fname)
            listOfData.append(temp_data)
            filename_origin.append(os.path.split(fname)[-1])
        img_data = np.array(listOfData)
        # find the center and crop it to new data
        col_pos = np.arange(0, col_size) - round(col_size / 2) + col_shift
        row_pos = np.arange(0, row_size) - round(row_size / 2) + row_shift
        img_crop = []
        for s_image in img_data:
            # s_image = s_image[round(s_image.shape[0]/2)-1000:round(s_image.shape[0]/2)+1000, round(s_image.shape[1]/2)-1000:round(s_image.shape[1]/2)+1000]
            background = np.mean(s_image[0:20, 0:20])
            print(np.mean(s_image[0:10, 0:10]))
            s_image = np.pad(s_image, (row_size, col_size), mode='constant')

            if np.amax(s_image) > 500:
                temp = s_image - background * 3
            else:
                temp = s_image - np.mean(s_image[0:20, 0:20])
            # print(temp.shape)
            temp = np.ones(temp.shape) * (temp >= 0)
            # plt.imshow(temp)
            # plt.show(block=False)
            # plt.pause(2)
            # plt.close()
            # plt.imshow(s_image)
            # plt.show()
            # plt.imshow(temp)
            # plt.show()
            if method_c == 'mess':
                M = sn.measurements.center_of_mass(temp)
                centroid_row = int(round(M[0]))
                centroid_col = int(round(M[1]))
            elif method_c == 'peak':
                M = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                centroid_row = M[0]
                centroid_col = M[1]
            elif method_c == 'image_center':
                centroid_row = int(round(temp.shape[0]/2))
                centroid_col = int(round(temp.shape[1]/2))
            else:
                print('wrong crop method')
                sys.exit()


            img_crop.append(
                s_image[centroid_row + row_pos[0]:centroid_row + row_pos[-1]+1,
                        centroid_col + col_pos[0]:centroid_col + col_pos[-1]+1])
        img_crop = np.array(img_crop, dtype='uint16')
        sub_folder = '/cropped/'

        if not os.path.exists(path_output + sub_folder):
            os.makedirs(path_output + sub_folder, exist_ok=True)

        for kk, s_image in enumerate(img_crop):

            tiff.imsave(
                path_output + sub_folder + filename_origin[kk],
                s_image)
            print('\033[32m' + 'MESSAGE: File ' + path_output + sub_folder +
                  filename_origin[kk]+
                  '   saved' + '\033[0m')
    
    elif len(sys.argv) == 5:
        '''
            if only three parameter, it's gui mode to get the folder path
            input:
                P_left_top:   the indice of the left top points
                P_right_bot:    the indice of the right bottom points

        '''
        start_row = int(sys.argv[1])
        end_row = int(sys.argv[3])
        start_col = int(sys.argv[2])
        end_col = int(sys.argv[4])

        print('\033[32m' + 'MESSAGE: select raw image folder' + '\033[0m')
        path_img = gui_load_data_directory('', 'load raw image folder')
        print('\033[32m' + 'MESSAGE: select output croped image folder' +
              '\033[0m')
        path_output = gui_load_data_directory('', 'output croped image folder')

        # read all the images under that folder
        listOfFiles = glob.glob(path_img + '/*.tif')
        if time_sort:
            listOfFiles.sort(key=os.path.getmtime)
        else:
            listOfFiles.sort()
        listOfData = []
        filename_origin = []
        for fname in listOfFiles:
            print('\033[32m' + 'MESSAGE: Open File ' + fname + '\033[0m')
            # temp_data = dxchange.read_tiff(fname)
            temp_data = tiff.imread(fname)
            listOfData.append(temp_data[start_row:end_row+1, start_col:end_col+1])
            filename_origin.append(os.path.split(fname)[-1])

        img_data = np.array(listOfData)

        # use the two points to crop the data and crop it to new data

        img_crop = img_data

        img_crop = np.array(img_crop, dtype='uint16')
        sub_folder = '/cropped/'

        if not os.path.exists(path_output + sub_folder):
            os.makedirs(path_output + sub_folder, exist_ok=True)

        for kk, s_image in enumerate(img_crop):

            tiff.imsave(
                path_output + sub_folder + filename_origin[kk],
                s_image)
            print('\033[32m' + 'MESSAGE: File ' + path_output + sub_folder +
                  filename_origin[kk],
                  '   saved' + '\033[0m')
        
        # # here to make the average of the images
        # sub_folder_average = '/average_crop/'
        # sub_folder_3d = '/3D_image/'
        # if not os.path.exists(path_output + sub_folder_average):
        #     os.makedirs(path_output + sub_folder_average, exist_ok=True)
        # if not os.path.exists(path_output + sub_folder_3d):
        #     os.makedirs(path_output + sub_folder_3d, exist_ok=True)
        # for kk in range(len(img_crop)//10):
        #     img_ave = np.mean(img_crop[kk*10:(kk+1)*10], axis=0)
        #     img_ave = np.array(img_ave, dtype='uint16')
        #     tiff.imsave(
        #         path_output + sub_folder_average + 'avg_image_' + str(kk) + '.tif',
        #         img_ave)
        #     print('\033[32m' + 'MESSAGE: File ' + path_output + sub_folder_average +
        #           'avg_image_' + str(kk) + '.tif'
        #           '   saved' + '\033[0m')
            
        #     XX, YY = np.meshgrid(range(img_ave.shape[0]), range(img_ave.shape[1]))
        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        #     surf = ax.plot_surface(XX, YY, img_ave, rstride=1, cstride=1, cmap=cm.get_cmap('hot'),
        #                             linewidth=0, antialiased=False)
        #     fig.colorbar(surf, shrink=0.5, aspect=5)
        #     plt.savefig(path_output + sub_folder_3d + '3D_image_' + str(kk) + '.png')
