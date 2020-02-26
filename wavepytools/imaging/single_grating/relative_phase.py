'''
    this program is used to calculate the relative phase change of different wavepy reconstruction results.
'''

import os
import sys
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def save_sdf_file(array, pixelsize=[1, 1], fname='output.sdf', extraHeader={}):
    '''
    Save an 2D array in the `Surface Data File Format (SDF)
    <https://physics.nist.gov/VSC/jsp/DataFormat.jsp#a>`_ , which can be
    viewed
    with the program `Gwyddion
    <http://gwyddion.net/documentation/user-guide-en/>`_ .
    It is also useful because it is a plain
    ASCII file
    Parameters
    ----------
    array: 2D ndarray
        data to be saved as *sdf*
    pixelsize: list
        list in the format [pixel_size_i, pixel_size_j]
    fname: str
        output file name
    extraHeader: dict
        dictionary with extra fields to be added to the header. Note that this
        extra header have not effect when using Gwyddion. It is used only for
        the asc file and when loaded by :py:func:`wavepy.utils.load_sdf`
        as *headerdic*.
    See Also
    --------
    :py:func:`wavepy.utils.load_sdf`
    '''

    if len(array.shape) != 2:
        prColor('ERROR: function save_sdf: array must be 2-dimensional', 'red')
        raise TypeError

    header = 'relative phase\n' + \
             'NumPoints\t=\t' + str(array.shape[1]) + '\n' + \
             'NumProfiles\t=\t' + str(array.shape[0]) + '\n' + \
             'Xscale\t=\t' + str(pixelsize[1]) + '\n' + \
             'Yscale\t=\t' + str(pixelsize[0]) + '\n' + \
             'Zscale\t=\t1\n' + \
             'Zresolution\t=\t0\n' + \
             'Compression\t=\t0\n' + \
             'DataType\t=\t7 \n' + \
             'CheckType\t=\t0\n' + \
             'NumDataSet\t=\t1\n' + \
             'NanPresent\t=\t0\n'

    for key in extraHeader.keys():
        header += key + '\t=\t' + extraHeader[key] + '\n'

    header += '*'

    if array.dtype == 'float64':
        fmt = '%1.8g'

    elif array.dtype == 'int64':
        fmt = '%d'

    else:
        fmt = '%f'

    np.savetxt(fname, array.flatten(), fmt=fmt, header=header, comments='')

    prColor('MESSAGE: ' + fname + ' saved!', 'green')


def load_sdf_file(fname, printHeader=False):
    '''
    Load an 2D array in the `Surface Data File Format (SDF)
    <https://physics.nist.gov/VSC/jsp/DataFormat.jsp#a>`_ . The SDF format
    is useful because it can be viewed with the program `Gwyddion
    <http://gwyddion.net/documentation/user-guide-en/>`_ .
    It is also useful because it is a plain
    ASCII file
    Parameters
    ----------
    fname: str
        output file name
    Returns
    -------
    array: 2D ndarray
        data loaded from the ``sdf`` file
    pixelsize: list
        list in the format [pixel_size_i, pixel_size_j]
    headerdic
        dictionary with the header
    Example
    -------
    >>> import wavepy.utils as wpu
    >>> data, pixelsize, headerdic = wpu.load_sdf('test_file.sdf')
    See Also
    --------
    :py:func:`wavepy.utils.save_sdf`
    '''

    with open(fname) as input_file:
        nline = 0
        header = ''
        if printHeader:
            print('########## HEADER from ' + fname)

        for line in input_file:
            nline += 1

            if printHeader:
                print(line, end='')

            if 'NumPoints' in line:
                xpoints = int(line.split('=')[-1])

            if 'NumProfiles' in line:
                ypoints = int(line.split('=')[-1])

            if 'Xscale' in line:
                xscale = float(line.split('=')[-1])

            if 'Yscale' in line:
                yscale = float(line.split('=')[-1])

            if 'Zscale' in line:
                zscale = float(line.split('=')[-1])

            if '*' in line:
                break
            else:
                header += line

    if printHeader:
        print('########## END HEADER from ' + fname)

    # Load data as numpy array
    data = np.loadtxt(fname, skiprows=nline)

    data = data.reshape(ypoints, xpoints)*zscale

    # Load header as a dictionary
    headerdic = {}
    header = header.replace('\t', '')

    for item in header.split('\n'):
        items = item.split('=')
        if len(items) > 1:
            headerdic[items[0]] = items[1]

    return data, [yscale, xscale], headerdic



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


if __name__ == "__main__":
    # define the path to the data folder
    file_path = gui_load_data_directory('', 'Path to the phase data folder')
    data_path = glob.glob(file_path+'/**/*_phase_*.sdf')
    prColor(data_path,'green')
    prColor(str(len(data_path))+' data are found', 'green')

    listOfData = []
    filename_origin = []
    for fname in data_path:
        prColor('MESSAGE: Open File ' + fname, 'green')
        temp_data, pixel_size, headerdic = load_sdf_file(fname)
        listOfData.append(temp_data)
        filename_origin.append(os.path.basename(os.path.dirname(fname)))
        prColor('subdir:' + filename_origin[-1], 'yellow')

    origin_data = np.array(listOfData)

    phase_data = origin_data - origin_data[0]

    y_axis = np.arange(phase_data.shape[1]) * pixel_size[0] * 1e3
    x_axis = np.arange(phase_data.shape[2]) * pixel_size[1] * 1e3


    YY, XX = np.meshgrid(y_axis, x_axis, indexing='ij')
    if not os.path.exists(file_path+'/processed/'):
        os.makedirs(file_path+'/processed/')
    for kk, phase in enumerate(phase_data):
        ax1 = plt.figure()
        im = plt.imshow(phase*1e9, cmap=cm.get_cmap('hot'))
        plt.colorbar(im, label='surface [nm]')
        plt.savefig(file_path+'/processed/'+filename_origin[kk]+'_2D.png')

        fig = plt.figure()
        ax2 = fig.gca(projection='3d')
        surf = ax2.plot_surface(XX, YY, phase*1e9, cmap=cm.get_cmap('hot'),
                       linewidth=0, antialiased=False)
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        fig.colorbar(surf, label='surface [nm]')
        plt.savefig(file_path+'/processed/'+filename_origin[kk]+'_3D.png')

        save_sdf_file(phase, pixel_size, file_path+'/processed/'+filename_origin[kk]+'_3D.sdf')


    if not os.path.exists(file_path+'/origin/'):
        os.makedirs(file_path+'/origin/')
    for kk, phase in enumerate(origin_data):
        ax1 = plt.figure()
        im = plt.imshow(phase*1e9, cmap=cm.get_cmap('hot'))
        plt.colorbar(im, label='surface [nm]')
        plt.savefig(file_path+'/origin/'+filename_origin[kk]+'_2D.png')

        fig = plt.figure()
        ax2 = fig.gca(projection='3d')
        surf = ax2.plot_surface(XX, YY, phase*1e9, cmap=cm.get_cmap('hot'),
                       linewidth=0, antialiased=False)
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        fig.colorbar(surf, label='surface [nm]')
        plt.savefig(file_path+'/origin/'+filename_origin[kk]+'_3D.png')

        