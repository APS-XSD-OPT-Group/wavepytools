'''
    here is code to do data processing after the wavefront reconstruction
'''

import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import scipy.constants as sc


def gui_load_data_file(directory='', title="File name with Data"):

    originalDir = os.getcwd()

    root = tk.Tk(title)
    # root.withdraw()
    fname1 = filedialog.askopenfilenames()

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1

    os.chdir(originalDir)

    return fname_last


def load_csv_new(path_cap):

    # Load data as numpy array
    data = np.loadtxt(path_cap, delimiter=',', encoding='utf-8-sig', skiprows=1)
    return data

# **************************************************************************
'''
    this part is for the line profile analysis.
    put all the line profiles together and calculate
    the curvature and radius
    and the rms and pv value
'''
energy = 14e3
wavelength = sc.value('inverse meter-electron volt relationship') / energy
file_list = gui_load_data_file('', 'line profile data')
listOfData = []
filename_origin = []
for fname in file_list:
    print('\033[32m' + 'MESSAGE: Open File ' + fname + '\033[0m')
    temp_data = load_csv_new(fname)
    listOfData.append(np.array(temp_data))
    filename_origin.append(os.path.split(fname)[-1].split('_'))

P_para = filename_origin

profile_data = np.array(listOfData)
if profile_data[0].shape[1] > 2:
    n_col = 3
else:
    n_col = 1
n_col = 2
# find the minimal range
x_min = profile_data[0][0][0]
x_max = profile_data[0][-1][0]
# x_min = -0.00011
# x_max = 0.00011

fig, ax = plt.subplots()
# fig, ax1 = plt.subplots()
for kk, data in enumerate(profile_data):

    x_axis = data[1:-1, 0]
    y_axis = data[1:-1, n_col]
    x_res = x_axis[(x_axis > x_min) & (x_axis < x_max)]
    y_res = y_axis[(x_axis > x_min) & (x_axis < x_max)]

    # start fitting
    p_fit = np.polyfit(x_axis, y_axis, 4)
    fit_v = lambda x: p_fit[2] * x ** 2 + p_fit[0] * x ** 4 + p_fit[1] * x ** 3
    p_fit2 = np.polyfit(x_axis, y_axis, 2)
    fit_res = lambda x: p_fit2[-3] * x ** 2 + p_fit2[-2] * x ** 1 + p_fit2[-1] 
    x_new = np.linspace(x_min, x_max, 100)
    y_fit = fit_v(x_new)
    y_new = y_fit - y_fit[0]

    y_res = y_res - fit_res(x_res)
    # y_res = (y_res - y_res[0])/wavelength
    y_res = (y_res - np.mean(y_res))/wavelength

    PV = np.max(y_res) - np.min(y_res)
    RMS = np.std(y_res)

    focus = -0.5 / p_fit[-3]
    # focus = 1/ (wavelength / 2 / np.pi *p_fit[-2])
    print('focal length: {} meter'.format(focus))

    # name = 'P1:' + P_para[kk][1] + '; P2:' + P_para[kk][3][0:-4] + '; R: ' + '{:.2f}'.format(focus) + 'm'
    # ax.plot(x_new*1e6, y_new/wavelength, label=name)
    # # ax.plot(x_axis*1e6, y_axis/wavelength, label=name)

    # plt.xlabel('vertical postion ($\mu$m)')
    # plt.ylabel('wavefront ($\lambda$)')

    # name1 = 'P1:' + P_para[kk][1] + '; P2:' + P_para[kk][3][0:-4] + '; RMS:' + '{:.2f}'.format(RMS) + '$\lambda$'
    name1 = P_para[kk][0][0:-4] + '; RMS:' + '{:.2f}'.format(RMS) + '$\lambda$'
    ax.plot(x_res*1e6, y_res, label=name1)

    plt.xlabel('vertical postion ($\mu$m)')
    plt.ylabel('wavefront error ($\lambda$)')


ax.legend(fontsize = 'x-small')
# ax1.legend()
plt.show()
# plt.show()

# **************************************************************************

