# -*- coding: utf-8 -*-  #
"""
Created on %(date)s

@author: %(username)s
"""

# %%% imports cell
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from labellines import labelLine, labelLines


# %%

import wavepy.utils as wpu
from wavepy.utils import easyqt
import glob


from scipy.optimize import curve_fit
from scipy.interpolate import interp1d



wpu._mpl_settings_4_nice_graphs(otheroptions={'lines.linewidth':2})


# %%
dirName = easyqt.get_directory_name()


print(dirName)
listOfFiles = glob.glob(dirName + '/*.csv')
listOfFiles.sort()
n_files = len(listOfFiles)



# %% Load data

listOfArrays = []
listOfShapes = []

for fname in listOfFiles:
    fileContent = wpu.load_csv_file(fname)
    listOfArrays.append(fileContent[0])

    listOfShapes.append(np.shape(fileContent[0]))

headers = fileContent[1]

# %%

plt.figure(figsize=(12,8))

for data, fname in zip(listOfArrays, listOfFiles):

    label = fname.rsplit('/', 1)[1].split('.')[0]

    plt.plot(data[:, 0]*1e6, data[:, 1], 'o', label=label)

    pfit = np.polyfit(data[:, 0], data[:, 1], 1)

    plt.plot(data[:, 0]*1e6, pfit[0]*data[:, 0] + pfit[1], '--', label=label)


plt.ylabel('WF ' + headers[2])
plt.xlabel('[µm]')
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)
plt.show(block=True)


# %%

plt.figure(figsize=(12,8))

listOfDPC_removed = []

for data, fname in zip(listOfArrays, listOfFiles):

    label = fname.rsplit('/', 1)[1].split('.')[0]

    pfit = np.polyfit(data[:, 0], data[:, 1], 1)

    dpc_removed = data[:, 1] - (pfit[0]*data[:, 0] + pfit[1])

    plt.plot(data[:, 0]*1e6, dpc_removed, '-o', label=label)

    listOfDPC_removed.append(dpc_removed)


plt.ylabel('WF ' + headers[2])
plt.xlabel('[µm]')
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show(block=True)
print('MESSAGE: Saving ' + figname)


# %%

wavelength = wpu.hc/12.4e3

plt.figure(figsize=(12,8))


for data, fname in zip(listOfArrays, listOfFiles):

    #    plt.figure(figsize=(12,8))
    pixelSize = data[1, 0] - data[0, 0]


    label = fname.rsplit('/', 1)[1].split('.')[0]

    integrated = -1/2/np.pi*wavelength*(np.cumsum(data[:, 1] - np.mean(data[:, 1]))*(pixelSize))


    plt.plot(data[:, 0]*1e6, integrated, 'o', label=label)



plt.ylabel('WF [m]')
plt.xlabel('[µm]')
plt.ylim((-1e-9,2e-9))
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/integrated', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)

plt.close('all')

# %%




plt.figure(figsize=(12,8))


for data, dpc_removed, fname in zip(listOfArrays, listOfDPC_removed, listOfFiles):

    #    plt.figure(figsize=(12,8))
    pixelSize = data[1, 0] - data[0, 0]


    label = fname.rsplit('/', 1)[1].split('.')[0]

    integrated = -1/2/np.pi*wavelength*(np.cumsum(dpc_removed - np.mean(dpc_removed))*(pixelSize))


    plt.plot(data[:, 0]*1e6, integrated, 'o', label=label)



plt.ylabel('WF [m]')
plt.xlabel('[µm]')
plt.ylim((-1.5e-10,1.5e-10))
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/removed_2nd_order_integrated', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)

plt.show(block=True)


plt.close('all')





# %% OLD
exit()

# %% define what to do, use of ini file

if 'Height' in headers[2]:
    what2do = 'Height response function'
else:

    what2do = wpu.get_from_ini_file(inifname, 'Parameters', 'what2do')

    if 'DPC' in what2do:
        choices = ['DPC response function', 'Curvature response function from diff data']
    else:
        choices = ['Curvature response function from diff data', 'DPC response function']

    what2do = easyqt.get_choice('Pick one', choices=choices)

wpu.set_at_ini_file(inifname, 'Parameters', 'what2do', what2do)

# %%

listInterpFunc = []

npoints_interp = 460
xnew = np.linspace(listOfArrays[0][0, 0], listOfArrays[0][-1, 0], npoints_interp)

# %%

from scipy.ndimage import gaussian_filter1d

if 'Curvature' in what2do:

    for data in listOfArrays:
        data[:, 1] = gaussian_filter1d(data[:, 1], 10)

# %%


plt.figure(figsize=(12,8))
previous_data = 0.0

for data, fname in zip(listOfArrays, listOfFiles):

    f = interp1d(data[:, 0], data[:, 1], kind='cubic')

    listInterpFunc.append(f)

    label = fname.rsplit('/', 1)[1].split('.')[0]

    plt.plot(data[:, 0]*1e6, data[:, 1], 'o', label=label)
    #    plt.plot(xnew, f(xnew), '-')

plt.ylabel('WF ' + headers[2])
plt.xlabel('[µm]')
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)


plt.show(block=False)

# %%


plt.figure(figsize=(12,8))

for data, fname in zip(listOfArrays, listOfFiles):

    f = interp1d(data[:, 0], data[:, 1], kind='cubic')

    listInterpFunc.append(f)

    label = fname.rsplit('/', 1)[1].split('.')[0]

    plt.plot(data[:, 0]*1e6, data[:, 1], '-', label=label)
    #    plt.plot(xnew, f(xnew), '-')

if n_files < 20:
    labelLines(plt.gca().get_lines(),align=False,fontsize=14)

plt.ylabel('WF ' + headers[2])
plt.xlabel('[µm]')
plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)


plt.show()


# %%
plt.figure(figsize=(12,8))


bufferdata = np.zeros(n_files)

timeStep = 10
xIndex2plot = np.size(data[:, 1])//2
for i, data in enumerate(listOfArrays):

    bufferdata[i] = data[xIndex2plot, 1]


foo = np.linspace(0, n_files*timeStep, n_files)
plt.plot(foo, bufferdata, '-o')

plt.title(what2do + ' at x = {:.2f} µm'.format(data[xIndex2plot, 0]*1e6//1))
plt.xlabel('Times [s]')
plt.ylabel('WF [m]')

figname = wpu.get_unique_filename(dirName + '/time_scan', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)

plt.show()


# %%

exit()

# %% Curvature

if 'Curvature' in what2do:

    listInterpFunc = []

    # curvature calculation and spline

    for data in listOfArrays:

        data_tmp = -1/2/np.pi*wavelength*np.diff(data[:, 1])/np.mean(np.diff(data[:, 0]))

        data[:, 1] = np.pad(data_tmp, (0 ,1), 'edge')

        f = interp1d(data[:, 0], data[:, 1], kind='cubic')

        plt.plot(data[:, 0]*1e6, data[:, 1], 'o', label=label + ' data')
        plt.plot(xnew, f(xnew), '-')

        listInterpFunc.append(f)

    plt.xlabel('[µm]')
    plt.title('Curvature [1/m] at {:.1f} KeV'.format(phenergy*1e3))
    plt.legend(loc='best', fontsize='x-small')
    plt.show()

    # %%
    #    for_csv = [xnew]
    #    for f_j in listInterpFunc:
    #        for_csv.append(f_j(xnew))
    #
    #    wpu.save_csv_file(for_csv, 'curv.csv')

# %%



    # %% plot curvature


#    fittedFunction = []
#
#    plt.figure()
#    for data, fname, f  in zip(listOfArrays, listOfFiles, listInterpFunc):
#        label = fname.rsplit('/', 1)[1].split('.')[0]
#        #        plt.plot(data[:, 0], data[:, 1], 'o', label=label + ' data')
#        #        plt.plot(xnew, f(xnew), '-')
#
#        popt, pcov = curve_fit(_gaussian_dist,
#                               data[:, 0],
#                               data[:, 1],
#                               [data[np.argmax(data[:, 1]), 0], 1e-4])
#
#        fitted = _gaussian_dist(xnew, popt[0], popt[1])
#
#        fittedFunction.append(fitted)
#        plt.plot(xnew, fitted, '-')
#
#    plt.title('Curvature [1/m]')
#    plt.legend(loc='best', fontsize='x-small')
#    plt.show()


# %%

M_matrix = np.zeros((npoints_interp, n_files))

for j in range(n_files):

    f_j = listInterpFunc[j]

    M_matrix[:, j] = f_j(xnew)/voltage4response

# %%
plt.figure()
plt.imshow(M_matrix, aspect='auto', origin='upper')
plt.title('M Matrix - ' + what2do)
plt.show()

# %%
    #
    ##M_matrix_plus = np.linalg.inv(M_matrix.T @ M_matrix) @ M_matrix.T
    #
    #M_inverse = np.linalg.pinv(M_matrix)
    #
    ##err = M_matrix_plus - M_inverse
    #
    ## %%
    #plt.figure()
    #plt.imshow(M_inverse, aspect='auto', origin='upper')
    #plt.title('M_inverse - ' + what2do)
    #plt.show()
    #
    ## %%
    #
    #jj, ii = np.mgrid[0:np.shape(M_inverse)[0], 0:np.shape(M_inverse)[1]]
    ## %%
    #
    #wpu.plot_profile(ii, jj, M_inverse, title= 'M_inverse - '+what2do)
    ##np.save('M_inverse_Jtec_mirror.npy', M_inverse)
    #
    ##foo = np.load('M_inverse_Jtec_mirror_DPC.npy')

# %% Target
#target = np.cos(xnew/8e-4*2*np.pi*3)*.2e-9

Radius = 2.0
target = Radius-np.sqrt(Radius**2-(xnew-0.0)**2)

dpc_target = np.diff(target)/np.mean(np.diff(xnew))/(-1/2/np.pi*wavelength)
curv_target = np.diff(dpc_target)/np.mean(np.diff(xnew))*(-1/2/np.pi*wavelength)

dpc_target = np.pad(dpc_target, (0, 1), 'edge')
curv_target = np.pad(curv_target, 1, 'edge')

# %%
if False:
    plt.figure()
    plt.plot(xnew, target*1e9)
    plt.title('Target')
    plt.xlabel(r'y [um]')
    plt.ylabel(r'height [nm]')
    plt.show()

    # target DPC

    plt.figure()
    plt.plot(xnew, dpc_target)
    plt.title('DPC Target')
    plt.show()

    # target curv

    plt.figure()
    plt.plot(xnew, curv_target)
    plt.title('Curvature Target')
    plt.show()


# %%
from scipy.optimize import lsq_linear

if 'Height' in what2do:

    res = lsq_linear(M_matrix, target, bounds=(bound_bottom, bound_top))

    #    voltage = M_inverse @ target
elif 'DPC' in what2do:

    res = lsq_linear(M_matrix, dpc_target, bounds=(bound_bottom, bound_top))
    #    voltage = M_inverse @ dpc_target
elif 'Curvature' in what2do:

    res = lsq_linear(M_matrix, curv_target, bounds=(bound_bottom, bound_top))
    #    voltage = M_inverse @ curv_target

voltage = res.x
voltage = voltage + base_voltage

# %%
plt.figure()
plt.plot(xnew, res.fun*1e9)
plt.title('Residual Target')
plt.xlabel(r'y [um]')
plt.ylabel(r'height [nm]')
plt.show()

# %%

for i, volts in enumerate(voltage):
    print('Channel {0}: {1:.3f} Volts'.format(i, volts))

for i, volts in enumerate(voltage):
    print('Channel {0}: {1:.3f} Volts'.format(i, volts - base_voltage))

# %%
plt.figure()
plt.plot(range(1, np.size(voltage) + 1),voltage, '-o')
plt.xlabel('Channel #')
plt.ylabel('Voltage [V]')
plt.savefig('voltage_' + what2do.replace(' ', '_') + '.png')
plt.show()

# %%



#if 'Height' in what2do:
#elif 'DPC' in what2do:
#elif 'Curvature [1/m]' in what2do:

# %%


wpu.save_sdf_file(M_matrix, fname='M_matrix_' + what2do.replace(' ', '_') + '.sdf')
wpu.save_sdf_file(M_inverse, fname='M_inverse_' + what2do.replace(' ', '_') + '.sdf')

wpu.save_csv_file([np.arange(1, np.size(voltage) + 1), voltage],
                  fname='voltage_' + what2do.replace(' ', '_') + '.csv',
                  headerList=['Channel', 'Voltage [V]'])


