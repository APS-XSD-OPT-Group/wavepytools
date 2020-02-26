'''
    this function provides a tool to image a series of 2D image in a stack. the image type is tiff
    for terminal mode, here is how to use it:
'''

import os
import sys
import numpy as np
from numpy import ogrid
# from mayavi.mlab import contour3d
import tifffile as tiff
import glob
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog 
from skimage.transform import resize
import scipy.io
from tkinter import Tcl

# import cv2

def gui_load_data_directory(directory='', title="File name with Data"):

    originalDir = os.getcwd()
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


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def plot_cube(cube, angle=320, IMG_DIM=100):
    
    cube = normalize(cube)
    viridis = cm.get_cmap('viridis')
    facecolors = viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    Z_dim, X_dim, Y_dim = cube.shape
    filled = facecolors[:,:,:,-1] > 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=X_dim*2)
    ax.set_ylim(top=Y_dim*2)
    ax.set_zlim(top=Z_dim*2)
    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()


if __name__ == "__main__":
    # get the images folder path
    path_images = gui_load_data_directory('', 'images folder')
    # path_images = '/Users/zhiqiao/Documents/Xray_wavefront_data/Zahir_Aug2019/CRL_D/focus_105p5mm_165p5mm_0p5mmstep/average_crop'
    
    # print(path_images)
    # read all the images
    # read all the images under that folder
    listOfFiles = glob.glob(path_images + '/*.tif')
    time_sort = False
    if time_sort:
        listOfFiles.sort(key=os.path.getmtime)
    else:
        listOfFiles = Tcl().call('lsort', '-dict', listOfFiles)
        # listOfFiles.sort()
    listOfData = []
    for fname in listOfFiles:
        print('\033[32m' + 'MESSAGE: read File ' + fname + '\033[0m')
        temp_data = tiff.imread(fname)
        listOfData.append(temp_data)

    img_data = np.array(listOfData)

    # use the 1/e^2 edge as the beam spot area
    n_image, row_size, col_size = img_data.shape
    img_width = np.zeros(img_data.shape)
    img_FWHM = np.zeros((2, img_data.shape[0]))
    img_sigma = np.zeros((2, img_data.shape[0]))

    norm_img = lambda img: (img - np.amin(img))/(np.amax(img)-np.amin(img))

    Y, X = np.meshgrid(range(row_size), range(col_size))
    IMG_DIM = 50
    img_resize = np.zeros((img_data.shape[0], IMG_DIM, IMG_DIM))

    for kk, s_image in enumerate(img_data):
        # s_image = norm_img(s_image)
        img_width[kk] = s_image * (s_image > np.amax(s_image) / 2)
        img_resize[kk] = resize(img_width[kk], (IMG_DIM, IMG_DIM), mode='constant')

        # calculate the image FWHM width
        Y_profile_FWHM = np.sum(img_width[kk], axis=1)
        X_profile_FWHM = np.sum(img_width[kk], axis=0)
        Y_profile = np.sum(s_image, axis=1)
        X_profile = np.sum(s_image, axis=0)
        Y_axis = (np.arange(len(Y_profile)) - np.round(len(Y_profile)/2)) * 0.65e-6
        X_axis = (np.arange(len(X_profile)) - np.round(len(X_profile)/2)) * 0.65e-6
        img_FWHM[0][kk] = np.sum(Y_profile_FWHM>0) * 0.65e-6
        img_FWHM[1][kk] = np.sum(X_profile_FWHM>0) * 0.65e-6

        #  caulcate the sigma beam width
        Y_average = np.sum(Y_profile * Y_axis) / np.sum(Y_profile)
        X_average = np.sum(X_profile * X_axis) / np.sum(X_profile)
        img_sigma[0][kk] = np.sqrt(np.sum(Y_profile*(Y_axis-Y_average)**2)/np.sum(Y_profile))
        img_sigma[1][kk] = np.sqrt(np.sum(X_profile*(X_axis-X_average)**2)/np.sum(Y_profile))

        print(kk)
        # # select the region in the 1/e^2 beam spot
        # img_width[kk] = 250 * (s_image > np.amax(s_image)/2)
        # # img_width[kk] = norm_img(img_width[kk])
        # img_edge = cv2.Canny(img_width[kk].astype(np.uint8), 100, 255)
        # indices = np.where(img_edge != [0])
        # # coordinates = zip(indices[0], indices[1])
        # Z[indices] += 2.5 * (kk-1)
        # plt.imshow(img_edge)
        # plt.show()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.get_cmap('jet'),
    #                    linewidth=0, antialiased=False)
    # plt.show()

    '''
        here the code generate the x-z and y-z side view of the 3d volume image.
        then the 3d data is saved as the matfile,which can be processed by matlab with volume viewer.
    '''

    img_xz = np.sum(img_width,axis=1)
    img_yz = np.sum(img_width,axis=2)
    path_save = '/output/'
    if not os.path.exists(path_images + path_save):
        os.makedirs(path_images + path_save, exist_ok=True)

    start_z_label = 105.5
    end_z_lable = 165.5
    start_x_label = -350 * 0.65
    end_x_lable = 350 * 0.65
    xtick = np.round(np.linspace(start_x_label, end_x_lable, 10))
    # ytick = np.linspace(start_z_label, end_z_lable, 10)
    ytick = np.linspace(1, img_xz.shape[0], 10)
    extent_s = [np.amin(xtick), np.amax(xtick), np.amin(ytick), np.amax(ytick)]
    fig1 = plt.figure()
    plt.imshow(np.flip(img_xz, axis=0) , cmap=cm.get_cmap('jet'),interpolation='spline16', extent=extent_s, aspect='auto')
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.xlabel('beam size (um)')
    plt.ylabel('image counts')
    plt.title('x-t view')
    plt.savefig(path_images + path_save+'View_xz.png', dpi=300, transparent=True)

    fig2 = plt.figure()
    plt.imshow(np.flip(img_yz, axis=0), cmap=cm.get_cmap('jet'), interpolation='spline16', extent=extent_s,aspect='auto')
    plt.xticks(xtick)
    plt.yticks(ytick)
    plt.title('y-t view')
    plt.xlabel('beam size (um)')
    plt.ylabel('image counts')
    plt.savefig(path_images + path_save+'View_yz.png', dpi=300, transparent=True)
    # plt.show()

    
    # z_axis = np.arange(start_z_label, end_z_lable+0.5, 0.5)
    z_axis = np.arange(0, img_xz.shape[0])
    fig3 = plt.figure()
    plt.plot(z_axis, img_FWHM[0]*1e6,'r--', label='Vertical')
    plt.plot(z_axis, img_FWHM[1]*1e6,'g^', label='Horizontal')
    plt.legend()
    plt.xlabel('image counts')
    plt.ylabel('FWHM beam size [um]')
    plt.title('FWHM beam size')
    plt.savefig(path_images + path_save+'FWHM_beamwidth.png', dpi=300, transparent=True)

    fig4 = plt.figure()
    plt.plot(z_axis, img_sigma[0]*1e6,'r--', label='Vertical')
    plt.plot(z_axis, img_sigma[1]*1e6,'g^', label='Horizontal')
    plt.legend()
    plt.xlabel('image counts')
    plt.ylabel('sigma beam size [um]')
    plt.title('Sigma beam size')
    plt.savefig(path_images + path_save+'Sigma_beamwidth.png', dpi=300, transparent=True)
    # plt.show(block=True)
    # plt.close()

    # scipy.io.savemat(path_images + path_save+'3d_data.mat', {'img_res':img_resize, 'img':img_data, 'img_filter':img_width})
    FWHM_min = np.amin(img_FWHM, axis=1)
    np.savetxt(path_images+path_save+'Beam_size.txt', FWHM_min*1e6, fmt='%1.8e', header='Beam FWHM minimum size [um]: vertical and horizontal')

    sys.exit()
    # plot_cube(img_resize[:10])

    # prepare some coordinates


    # # combine the objects into a single boolean array
    # # voxels = cube1 | cube2 | link
    # voxels = img_resize[:10] > 0.5

    # # set the colors of each object
    # colors = np.empty(voxels.shape, dtype=object)
    # colors[voxels] = 'red'
    # # colors[cube1] = 'blue'
    # # colors[cube2] = 'green'

    # # and plot everything
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')

    # plt.show()
        



