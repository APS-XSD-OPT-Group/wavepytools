#!/usr/bin/env python
"""
use the minimization method to get the response function and give the 
linear decomposition under certain constraints, like up, down, and max-min difference
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import csv
import tkinter as tk
from tkinter import filedialog
import glob
import scipy.constants as sconst
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import scipy.io as sio

'''
    the response function folder: 
        18 wavefront process folder, *_output
    target file:
        csv file, with first col for x, second col for y
        first row is header
    current file:
        csv file, the wavefront measured for current setting
        first row is header
'''


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

def gui_load_data_file(directory='', title="data file"):
    
    originalDir = os.getcwd()

    if directory != '':

        if os.path.isdir(directory):
            os.chdir(directory)
        else:
            prColor("WARNING: Directory " + directory + " doesn't exist.", 'red')
            prColor("MESSAGE: Using current working directory " + originalDir, 'yellow')
    
    root = tk.Tk(title)
    root.withdraw()
    fname1 = filedialog.askopenfile()
    # fname1 = easyqt.get_directory_name(title)

    if len(fname1) == 0:
        fname_last = None

    else:
        fname_last = fname1

    os.chdir(originalDir)

    return fname_last


def load_target_file(file_path):
    '''
    load the target data
    '''
    
    x_pos = []
    y_pos = []

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        skipline = 1
        x = []
        y = []
        for row in readCSV:
            if skipline > 0:
                skipline -= 1
                continue
            else:
                x.append(float(row[0]))
                y.append(float(row[3]))
    x_pos.append(x)
    y_pos.append(y)

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    return x_pos, y_pos

def load_current_file(file_path):
    '''
    load the target data
    '''
    
    x_pos = []
    y_pos = []

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        skipline = 1
        x = []
        y = []
        for row in readCSV:
            if skipline > 0:
                skipline -= 1
                continue
            else:
                x.append(float(row[0]))
                y.append(float(row[3]))
    x_pos.append(x)
    y_pos.append(y)

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    return x_pos, y_pos
    

def load_response_function(Folder_path, profilenumber):
    os.chdir(Folder_path)
    prColor('folder: '+os.getcwd(), 'green')
    # f_list = glob.glob('*_output/profiles/*_integrated_y_01.csv')
    f_list = glob.glob('ch*.csv')


    x_pos = []
    y_pos = []
    dataname = []

    for f_single in f_list:

        with open(f_single) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            skipline = 1
            x = []
            y = []
            for row in readCSV:
                if skipline > 0:
                    skipline -= 1
                    continue
                else:
                    x.append(float(row[0]))
                    y.append(float(row[profilenumber]))
        x_pos.append(x)
        y_pos.append(y)

        # dataname.append(os.path.basename(f_single)[0:11])
        dataname.append(os.path.basename(f_single))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    dataname = np.array(dataname)

    return x_pos, y_pos, dataname

def multi_target_optimization(m_matrix, target, bound_bottom, bound_top):
    '''
        add other constraints to get the linear decomposation results
        minimize:      beta * ||Ax - C||_2 + ||x-x_mean||_2 
    '''
    x0 = np.zeros(m_matrix.shape[1])
    beta = 1000
    bnds = []
    for bnds_up, bnds_low in zip(bound_top, bound_bottom):
        bnds.append([bnds_low, bnds_up])
    func_optimize = lambda x: beta * np.sum((m_matrix @ x - target)**2) / np.sum(target**2) \
                            + 1 * np.sum((x[0:18] - np.mean(x[0:18]))**2) / np.sum(x[0:18]**2) \
                                + 1 * np.sum((x[0:18] - np.mean(x[0:18]))**2) / np.sum(x[0:18]**2)

    res = minimize(func_optimize, x0, method='nelder-mead', bounds=bnds, options={'xtol': 1e-16, 'disp': True})

    res_profile = m_matrix @ res.x - target
    return res, res_profile


#base_voltage = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
base_voltage = [700] * 18
voltage4response = 100.00	#applied change in user unit, for bimorph, it is voltage, for bender, it may be the displacement of bender position

phenergy = 14200	#in eV
wavelength = sconst.value('inverse meter-electron volt relationship') / phenergy

profilenumber = 3	#profile index in the target and RF csv files

subtractnominal = 0	#subtract a norminal spherical wf with Radius below
Radius = -0.71


npoints_interp = 200	#number of points to interpolate both the target and the RF

Folder_RF = 'C:/Users/zqiao/Documents/xray_wavefront/Response/RF/'
target_file = 'C:/Users/zqiao/Documents/xray_wavefront/Response/target.csv'
# the wavefront measured with current state
# current_file = 'C:/Users/zqiao/Documents/xray_wavefront/Response/target.csv'
Folder_result = 'C:/Users/zqiao/Documents/xray_wavefront/Response/'
# if use target or not
Fit_to_target = True

if Folder_RF == '':
    Folder_RF = gui_load_data_directory('', title='Folder for the response function')
elif not os.path.isdir(Folder_RF):
    prColor('wrong folder path!', 'red')
    sys.exit()

# load the response function from the folder, where all the wavefront data is there
x_pos, y_pos, dataname = load_response_function(Folder_RF, profilenumber)
n_files = np.amin(x_pos.shape)

if n_files > 18:
    prColor('too many data for the response function', 'red')
    sys.exit()

# if current_file == '':
#     current_file = gui_load_data_file('', title='current measured wavefront')
# elif not os.path.isfile(current_file):
#     prColor('wrong current wavefront file path path!', 'red')
#     sys.exit()

if Fit_to_target and target_file == '':
    
    target_file = gui_load_data_file('', title='target file')

elif not os.path.isfile(target_file):
    prColor('wrong target file path!', 'red')
    sys.exit()

# get the data from the target file
# x_current, y_current = load_target_file(current_file)
# x_current = x_current[0]
# y_current = y_current[0] *1e-15

# get the data from the target file
x_target, y_target = load_target_file(target_file)
x_target = x_target[0]
y_target = y_target[0] *1e-15

plt.figure(figsize=(12, 8))

data_fitting = []

for x, y, fname in zip(x_pos, y_pos, dataname):
    f = interp1d(x, y, kind='cubic')
    data_fitting.append(f)

    plt.plot(x*1e6, y*1e9, 'o', label=fname)
    #    plt.plot(xnew, f(xnew), '-')

plt.ylabel('WF [nm]')
plt.xlabel('[µm]')
plt.legend(loc='best', fontsize='x-small')
plt.savefig(os.path.join(Folder_result, 'response_function.png'))
prColor('MESSAGE: response function saving', 'green')

plt.show(block=True)

xnew = np.linspace(np.amax(x_pos[:, 0]), np.amax(x_pos[:, -1]), npoints_interp)
m_matrix = np.zeros((npoints_interp, n_files + 2))

for j in range(n_files):

    f_j = data_fitting[j]

    m_matrix[:, j] = f_j(xnew)/voltage4response

m_matrix[:, -2] = np.ones(npoints_interp)  # piston term
m_matrix[:, -1] = xnew # tilt
#m_matrix[:, -1] = xnew**2 # second order

plt.figure()
plt.imshow(m_matrix[:,:-2], aspect='auto', origin='upper')
plt.title('M Matrix Height')
plt.savefig(os.path.join(Folder_result, 'response_function_18ch.png'))
plt.show(block=False)

plt.figure()
plt.imshow(m_matrix, aspect='auto', origin='upper')
plt.title('M Matrix - Height')
plt.savefig(os.path.join(Folder_result, 'response_function_20ch.png'))
plt.show()



#Radius = 1.828
if subtractnominal == 0:
	nominal = 0.0*xnew
else:	
	nominal = -(Radius-np.sqrt(Radius**2-(xnew-0.0)**2))


#get the current wavefront measurement

# f_current = interp1d(x_current, -y_current, kind='cubic')

# if xnew[-1] <= x_current[-1]:
#     current = f_current(xnew)
#     current -= nominal
# else:
#     current = xnew*0.0
#     current[np.where(np.abs(xnew)<x_current[-1])] = f_current(xnew[np.where(np.abs(xnew)<x_current[-1])])
#     current[np.where(np.abs(xnew)<x_current[-1])] -= nominal[np.where(np.abs(xnew)<x_current[-1])]	#This is the difference between the nominal sphere wf and the measured wf

#target is the negative of measured wavefront, this is to get flat wf

f_target = interp1d(x_target, -y_target, kind='cubic')

if xnew[-1] <= x_target[-1]:
    target = f_target(xnew)
    target -= nominal		
else:
    target = xnew*0.0
    target[np.where(np.abs(xnew)<x_target[-1])] = f_target(xnew[np.where(np.abs(xnew)<x_target[-1])])
    target[np.where(np.abs(xnew)<x_target[-1])] -= nominal[np.where(np.abs(xnew)<x_target[-1])]	#This is the difference between the nominal sphere wf and the measured wf



plt.figure()
plt.plot(xnew*1e6, target*1e9)
plt.title('Target_before cut ' +
            ', rms = {:.2f} pm'.format(np.std(target)*1e12))
plt.xlabel('y [um]')
plt.ylabel('height [nm]')
plt.savefig(os.path.join(Folder_result, 'response_function_target.png'))
plt.show(block=False)
#
## %%
#
#min_x = -340e-6
#max_x = 340e-6
#
#arg_min = np.argmin((xnew-min_x)**2)
#
#arg_max = np.argmin((xnew-max_x)**2)
#
#cut the wavefront and RF
#
'''
    to crop the wavefront and target to a small range
'''
arg_min = 50
arg_max = -50
#
m_matrix = m_matrix[arg_min:arg_max,:]
target = target[arg_min:arg_max]
xnew = xnew[arg_min:arg_max]
if True:
    plt.figure()
    plt.plot(xnew*1e6, target*1e9)
    #    plt.plot(temp_Data[:, 0]*1e6, temp_Data[:, 1]*1e9)
    plt.title('Target_cropped, ' + 
              ', rms = {:.2f} pm'.format(np.std(target)*1e12))
    plt.xlabel('y [um]')
    plt.ylabel('height [nm]')
    plt.show(block=False)


# remove 1st order

if True:
    pfit = np.polyfit(xnew, target, 1)
    fitted_func = pfit[0]*xnew + pfit[1]

    # %%
#    plt.figure(figsize=(12, 8))
#    plt.plot(xnew*1e6, target, '-')
#    plt.plot(xnew*1e6, fitted_func, '-')

#    plt.ylabel('WF ' + headers[-1])
#    plt.xlabel('[µm]')
#    plt.title('1st order polynomial Fit')
#    plt.show()

    target -= fitted_func
    
    plt.figure()
    plt.plot(xnew*1e6, target*1e9)
    #    plt.plot(temp_Data[:, 0]*1e6, temp_Data[:, 1]*1e9)
    plt.title('Target remove linear phase ' +
              ', rms = {:.2f} pm'.format(np.std(target)*1e12))
    plt.xlabel('y [um]')
    plt.ylabel('height [nm]')
    plt.savefig(os.path.join(Folder_result, 'response_function_target_nolinearPhase.png'))
    plt.show(block=False)

'''
    remove second phase from the target
'''
pfit = np.polyfit(xnew, target, 2)
bestfit2nd = pfit[0]*xnew**2 + pfit[1]*xnew + pfit[2]
target -= bestfit2nd

dpc_target = np.diff(target)/np.mean(np.diff(xnew))/(-1/2/np.pi*wavelength)
curv_target = np.diff(dpc_target)/np.mean(np.diff(xnew))*(-1/2/np.pi*wavelength)

dpc_target = np.pad(dpc_target, (0, 1), 'edge')
curv_target = np.pad(curv_target, 1, 'edge')

'''
    start the fitting to find the best setting
'''
from scipy.optimize import lsq_linear, least_squares

bound_all = 200.000
block_all = 0.001

bound_top = np.array([bound_all, bound_all, bound_all, bound_all,
                      bound_all, bound_all, bound_all, bound_all,
                      bound_all, bound_all, bound_all, bound_all,
                      bound_all, bound_all, bound_all, bound_all,
                      bound_all, bound_all,
                      1e20, 1e20])

bound_all = -200.00
block_all = -0.001

bound_bottom = np.array([bound_all, bound_all, bound_all, bound_all,
                         bound_all, bound_all, bound_all, bound_all,
                         bound_all, bound_all, bound_all, bound_all,
                         bound_all, bound_all, bound_all, bound_all,
                         bound_all, bound_all,
                         -1e20, -1e20])

'''
    use least square method, no constraint for the max change of the near coefficient
'''

# res = lsq_linear(m_matrix, dpc_target, bounds=(bound_bottom, bound_top),
#                      method='bvls', tol=1e-32, verbose=1, max_iter=1000)
# res_profile = res.fun
'''
    use minimzation and maximal change constraints
'''

res, res_profile = multi_target_optimization(m_matrix, target, bound_bottom, bound_top)

print('Status: {}'.format(res.status))

success = res.success
x_result = res.x


if success:
    print("Uha!")
else:
    print("FAIL!!!!")
voltage = x_result[:-2]
piston = x_result[-2]
tilt = x_result[-1]

print(x_result)

for i, fname in enumerate(dataname):
    prColor('voltage for channel {}: {}'.format(fname, voltage[i]), 'green')

prColor('piston: {:.4g} nm'.format(piston*1e9), 'green')
prColor('tilt: {:.4g} rad?'.format(tilt), 'green')




for nn in range(18):
    prColor('{:.1f}'.format(voltage[nn] + base_voltage[nn]), 'purple')
    
'''
    save the data
'''
filename = os.path.basename(target_file) + '_responseFunc_results.mat'
sio.savemat(os.path.join(Folder_result, filename), {'m_matrix': m_matrix, 'target_x': x_target, 'target_y': y_target, 'voltage': voltage})

# %%
voltage4plot = np.zeros(np.size(voltage)+2)
voltage4plot[:-2] = voltage

voltage4plot[-2] = piston
voltage4plot[-1] = tilt


finalSurface = m_matrix @ voltage4plot
plt.figure()
plt.plot(xnew*1e6, finalSurface*1e9)
plt.ylabel('Height [nm]')
plt.title('Surface Displacement, rms = {:.2f} pm'.format(np.std(finalSurface)))
plt.xlabel('y [um]')
plt.savefig(os.path.join(Folder_result, 'response_func_after_correct.png'))
plt.show(block=False)

plt.figure()
plt.plot(xnew*1e6, -(res_profile-np.mean(res_profile))*1e9, '-', label='Residual')
plt.plot(xnew*1e6, (target-np.mean(target))*1e9, '-', label='Target')
plt.ylabel('Height [nm]')

#plt.title('Residual, Correction: {:} V'.format(bound_top))

plt.title('Target, '
          ', rms = {:.2f} pm'.format(np.std(target)*1e12) +
          '\nResidual, rms = {:.2f} pm'.format(np.std(res_profile)*1e12))
plt.xlabel('y [um]')
plt.legend()
plt.savefig(os.path.join(Folder_result, 'response_func_residual.png'))
plt.show(block=False)

plt.figure()
plt.bar(range(1, np.size(voltage) + 1),voltage, width=1.0)
plt.xlabel('Channel #')
plt.ylabel('Voltage [V]')
plt.title('Final Voltage')
plt.savefig(os.path.join(Folder_result, 'response_func_voltage.png'))
plt.show()

