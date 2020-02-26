#!/usr/bin/env python
"""
Created on %(date)s

@author: %(username)s
"""

# %%% imports cell
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from labellines import labelLine, labelLines
import sys

# %%

import wavepy.utils as wpu
from wavepy.utils import easyqt
import glob

from scipy.interpolate import interp1d

def load_csv_new(path_cap):
    '''
    Here use the cap to wavesensor data transfer function to generate the same structure
    data file with the wavefront measurement curve. The file is generated in the dir_name path
    '''
    # Load data as numpy array
    data = np.loadtxt(path_cap, delimiter=',', encoding='utf-8-sig', skiprows=1)
    header = ['Height']
    comments = []
    return data, header, comments
    
def _gaussian_dist(x, sigma, xo):

    return 1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-xo)**2/2/sigma**2)

wpu._mpl_settings_4_nice_graphs(otheroptions={'lines.linewidth': 2})

#base_voltage = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
base_voltage = [643.6, 480.6, 386.2, 556.3, 365.9, 566.7, 393.9, 581.5, 475.5, 547.3, 476.6, 584.2, 481.5, 485.4, 646.4, 448.1, 503.4, 1000.0]
voltage4response = 200.00	#applied change in user unit, for bimorph, it is voltage, for bender, it may be the displacement of bender position

phenergy = 14000	#in eV
wavelength = wpu.hc/phenergy

profilenumber = 3	#profile index in the target and RF csv files

subtractnominal = 1	#subtract a norminal spherical wf with Radius below
Radius = -0.71


npoints_interp = 200	#number of points to interpolate both the target and the RF

# %%

if len(sys.argv) == 1:
    dirName = easyqt.get_directory_name(title='Select folder contains RF')
else:
    dirName = sys.argv[1]


inifname = '.response_function.ini'
defaults = wpu.load_ini_file(inifname)

if dirName == '':
    dirName = defaults['Files'].get('Folder with csv files')

else:
    wpu.set_at_ini_file(inifname,
                        'Files',
                        'Folder with csv files', dirName)

print(dirName)
listOfFiles = glob.glob(dirName + '/*.csv')
listOfFiles.sort()
n_files = len(listOfFiles)

# %% Load data

listOfArrays = []
listOfShapes = []

for fname in listOfFiles:
    wpu.print_blue('MESSAGE: Open File ' + fname)
    #fileContent = wpu.load_csv_file(fname)
    fileContent = load_csv_new(fname)
    listOfArrays.append(fileContent[0])

    listOfShapes.append(np.shape(fileContent[0]))

headers = fileContent[1]


# %%
for data, fname in zip(listOfArrays, listOfFiles):
    label = fname.rsplit('/', 1)[1].split('.')[0]
    print(label + ', rms value: {:.4f} nm'.format(np.std(data[:, profilenumber])*1e9))

# %% define what to do, use of ini file

if 'Height' in headers[-1]:
    what2do = 'Height response function'
else:
    what2do = wpu.get_from_ini_file(inifname, 'Parameters', 'what2do')

    if 'DPC' in what2do:
        choices = ['DPC response function', 'Curvature response function from diff data']
    else:
        choices = ['Curvature response function from diff data', 'DPC response function']

    what2do = easyqt.get_choice('Pick one', choices=choices)

wpu.set_at_ini_file(inifname, 'Parameters', 'what2do', what2do)

# %% Target


if len(sys.argv) > 2 or easyqt.get_yes_or_no('Do you want to load a target file?'):

    if len(sys.argv) > 2:
        targetName = sys.argv[2]
    else:
        targetName = easyqt.get_file_names(title='select target CSV')

    if targetName == []:
        targetName = defaults['Files'].get('target file')
    else:
        wpu.set_at_ini_file(inifname,
                            'Files',
                            'target file', targetName[0])

    #temp_Data = wpu.load_csv_file(targetName[0])[0]
    temp_Data = load_csv_new(targetName[0])[0]
    lim_xnew = np.min((np.abs(listOfArrays[0][0, 0]),
                       np.abs(listOfArrays[0][-1, 0]),
                       np.abs(temp_Data[0, 0]),
                       np.abs(temp_Data[-1, 0])))
else:
    temp_Data = None

    lim_xnew = np.min((np.abs(listOfArrays[0][0, 0]),
                       np.abs(listOfArrays[0][-1, 0])))

# %%


xnew = np.linspace(-lim_xnew, lim_xnew, npoints_interp)

# %%

#exit()


# %%

from scipy.ndimage import gaussian_filter1d

if 'Curvature' in what2do:

    for data in listOfArrays:
        data[:, profilenumber] = gaussian_filter1d(data[:, profilenumber], 50)

# %%

plt.figure(figsize=(12, 8))

listInterpFunc = []

for data, fname in zip(listOfArrays, listOfFiles):

    f = interp1d(data[:, 0], data[:, profilenumber], kind='cubic')
    listInterpFunc.append(f)

    label = fname.rsplit('/', 1)[1].split('.')[0]

    plt.plot(data[:, 0]*1e6, data[:, profilenumber], 'o', label=label)
    #    plt.plot(xnew, f(xnew), '-')

plt.ylabel('WF ' + headers[-1])
plt.xlabel('[µm]')

if n_files < 20:
    plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.show(block=True)

# %% Plot with labels inline

plt.figure(figsize=(12, 8))

for data, fname in zip(listOfArrays, listOfFiles):

    label = fname.rsplit('/', 1)[1].split('.')[0]
    plt.plot(data[:, 0]*1e6, data[:, profilenumber], '-', label=label)

if n_files < 20:
    labelLines(plt.gca().get_lines(),align=False,fontsize=14,
               xvals=(data[0, 0]*1e6, data[-1, 0]*1e6))

plt.ylabel('WF ' + headers[-1])
plt.xlabel('[µm]')
#plt.legend(loc='best', fontsize='x-small')

figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
print('MESSAGE: Saving ' + figname)


xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
plt.show(block=True)

plt.show()


# %% Animation

timeStep = 1
t = 0

if False:

    plt.figure(figsize=(12, 8))

    plt.ylabel('WF ' + headers[-1])
    plt.xlabel('[µm]')

    for data, fname in zip(listOfArrays, listOfFiles):

        label = fname.rsplit('/', 1)[1].split('.')[0]

        plt.plot(data[:, 0]*1e6, data[:, profilenumber], '-k', label=label)
        #        plt.title('t = {:d}s'.format(t))
        plt.title(label)
        t += timeStep

        plt.xlim(xlim)
        plt.ylim(ylim)
        figname = wpu.get_unique_filename(dirName + '/anim_respons_func', 'png', width=3)
        plt.savefig(figname)
        print('MESSAGE: Saving ' + figname)

        del plt.gca().lines[0]

    plt.close('all')


# %% Time plot

if False:

    plt.figure(figsize=(12, 8))

    bufferdata = np.zeros(n_files)

    xIndex2plot = np.size(data[:, 0])//2
    xIndex2plot = np.argmin((data[:, 0]-0.0e-6)**2)
    for i, data in enumerate(listOfArrays):

        bufferdata[i] = data[xIndex2plot, 1]

        #        bufferdata[i] = np.ptp(data[:, profilenumber])

    foo = np.linspace(0, (n_files-1)*timeStep, n_files)
    plt.plot(foo, bufferdata, '-o')

    plt.title(what2do + ' at x = {:.0f} µm'.format(data[xIndex2plot, 0]*1e6//1))

    #    plt.title(what2do + ', PV')
    plt.xlabel('Times [s]')
    plt.ylabel('WF [m]')

    figname = wpu.get_unique_filename(dirName + '/resp_func_time_scan', 'png')
    plt.savefig(figname)
    print('MESSAGE: Saving ' + figname)

    plt.show()


# %%

    wpu.save_csv_file([foo, bufferdata],
                      fname=wpu.get_unique_filename(dirName + '/resp_func_time_scan', 'dat'),
                      headerList=['Time',
                                  what2do +
                                  ' at x = {:.0f} µm'.format(data[xIndex2plot, 0]*1e6//1)])

# %% Curvature

if 'Curvature' in what2do:

    listInterpFunc = []

    # curvature calculation and spline



    listOfArrays_tmp = []


    for data, fname in zip(listOfArrays, listOfFiles):

        plt.figure(figsize=(12, 8))


        label = fname.rsplit('/', 1)[1].split('.')[0]

        data_tmp = -1/2/np.pi*wavelength*np.diff(data[:, profilenumber])/np.mean(np.diff(data[:, 0]))
        data_tmp = np.pad(data_tmp, (0 ,1), 'edge')

        f = interp1d(data[:, 0], data_tmp, kind='cubic')

        plt.plot(data[:, 0]*1e6, data_tmp, 'o', label=label + ' data')
        #        plt.plot(xnew, f(xnew), '-')

        listInterpFunc.append(f)

        plt.xlabel('[µm]')
        plt.title('Curvature [1/m] at {:.1f} KeV'.format(phenergy*1e3))
        plt.legend(loc='best', fontsize='x-small')

        figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
        plt.savefig(figname)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.show(block=True)



# %% remove 2nd order

if False:
    listOfArrays_Buff = np.copy(listOfArrays)

    # %%
    plt.figure(figsize=(12, 8))

    for data, fname in zip(listOfArrays, listOfFiles):

        pfit = np.polyfit(data[:, 0], data[:, profilenumber], 2)

        label = fname.rsplit('/', 1)[1].split('.')[0]

        plt.plot(data[:, 0]*1e6, data[:, profilenumber], '-o', label=label)

        fitted_func = pfit[0]*data[:, 0]**2 + pfit[1]*data[:, 0] + pfit[2]
        plt.plot(data[:, 0]*1e6, fitted_func, '--c')

        data[:, profilenumber] -= fitted_func  # make change permanent

    plt.ylabel('WF ' + headers[-1])
    plt.xlabel('[µm]')
    plt.title('2nd order polynomial Fit')
    #plt.legend(loc='best', fontsize='x-small')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
    plt.show()

    # %%
    plt.figure(figsize=(12, 8))

    for data, fname in zip(listOfArrays, listOfFiles):

        label = fname.rsplit('/', 1)[1].split('.')[0]
        plt.plot(data[:, 0]*1e6, data[:, profilenumber], '-', label=label)

    plt.title('Residual from 2nd order polynomial')

    if n_files < 20:
        labelLines(plt.gca().get_lines(),align=False, fontsize=14,
                   xvals=(data[0, 0]*1e6, data[-1, 0]*1e6))

    #    plt.legend(loc='best', fontsize='x-small')

    plt.ylabel('WF ' + headers[-1])
    plt.xlabel('[µm]')

    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
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
#        #        plt.plot(data[:, 0], data[:, profilenumber], 'o', label=label + ' data')
#        #        plt.plot(xnew, f(xnew), '-')
#
#        popt, pcov = curve_fit(_gaussian_dist,
#                               data[:, 0],
#                               data[:, profilenumber],
#                               [data[np.argmax(data[:, profilenumber]), 0], 1e-4])
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

m_matrix = np.zeros((npoints_interp, n_files + 2))

for j in range(n_files):

    f_j = listInterpFunc[j]

    m_matrix[:, j] = f_j(xnew)/voltage4response

m_matrix[:, -2] = np.ones(npoints_interp)  # piston term
m_matrix[:, -1] = xnew # tilt
#m_matrix[:, -1] = xnew**2 # second order

# %%
plt.figure()
plt.imshow(m_matrix[:,:-2], aspect='auto', origin='upper')
plt.title('M Matrix - ' + what2do)
figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.imshow(m_matrix, aspect='auto', origin='upper')
plt.title('M Matrix - ' + what2do)
figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show()

# %%


if temp_Data is None:
    exit()

# %%

#exit()

#Radius = 1.828
if subtractnominal == 0:
	nominal = 0.0*xnew
else:	
	nominal = -(Radius-np.sqrt(Radius**2-(xnew-0.0)**2))

# %%

target = -temp_Data[:, 1]	#target is the negative of measured wavefront, this is to get flat wf


f_target = interp1d(temp_Data[:, 0], target, kind='cubic')

if xnew[-1] <= temp_Data[-1, 0]:
    target = f_target(xnew)
    target -= nominal		
else:
    target = xnew*0.0
    target[np.where(np.abs(xnew)<temp_Data[-1, 0])] = f_target(xnew[np.where(np.abs(xnew)<temp_Data[-1, 0])])
    target[np.where(np.abs(xnew)<temp_Data[-1, 0])] -= nominal[np.where(np.abs(xnew)<temp_Data[-1, 0])]	#This is the difference between the nominal sphere wf and the measured wf


# %%
if True:
    plt.figure()
    plt.plot(xnew*1e6, target*1e9)
    #    plt.plot(temp_Data[:, 0]*1e6, temp_Data[:, 1]*1e9)
    plt.title('Target_before cut, ' + targetName[0].rsplit('/', 1)[1] +
              ', rms = {:.2f} pm'.format(np.std(target)*1e12))
    plt.xlabel(r'y [um]')
    plt.ylabel(r'height [nm]')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
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
arg_min = 50
arg_max = -30
#
m_matrix = m_matrix[arg_min:arg_max,:]
target = target[arg_min:arg_max]
xnew = xnew[arg_min:arg_max]
if True:
    plt.figure()
    plt.plot(xnew*1e6, target*1e9)
    #    plt.plot(temp_Data[:, 0]*1e6, temp_Data[:, 1]*1e9)
    plt.title('Target_cropped, ' + targetName[0].rsplit('/', 1)[1] +
              ', rms = {:.2f} pm'.format(np.std(target)*1e12))
    plt.xlabel(r'y [um]')
    plt.ylabel(r'height [nm]')
    plt.show(block=False)


# %% remove 1st order

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

 
#pfit = np.polyfit(xnew, target, 2)
#bestfit2nd = pfit[0]*xnew**2 + pfit[1]*xnew + pfit[2]
#target -= bestfit2nd

dpc_target = np.diff(target)/np.mean(np.diff(xnew))/(-1/2/np.pi*wavelength)
curv_target = np.diff(dpc_target)/np.mean(np.diff(xnew))*(-1/2/np.pi*wavelength)

dpc_target = np.pad(dpc_target, (0, 1), 'edge')
curv_target = np.pad(curv_target, 1, 'edge')



# %%
if True:
    plt.figure()
    plt.plot(xnew*1e6, target*1e9)
    #    plt.plot(temp_Data[:, 0]*1e6, temp_Data[:, 1]*1e9)
    plt.title('Target, ' + targetName[0].rsplit('/', 1)[1] +
              ', rms = {:.2f} pm'.format(np.std(target)*1e12))
    plt.xlabel(r'y [um]')
    plt.ylabel(r'height [nm]')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
    plt.show(block=False)

if False:
    # target DPC

    plt.figure()
    plt.plot(xnew*1e6, dpc_target)
    plt.xlabel(r'y [um]')
    plt.title('DPC Target')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
    plt.show(block=False)

    # target curv

    plt.figure()
    plt.plot(xnew*1e6, curv_target)
    plt.xlabel(r'y [um]')
    plt.title('Curvature Target')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
    plt.show(block=True)

# %%

#exit()

# %%
from scipy.optimize import lsq_linear, least_squares


bound_all = 100.000
block_all = 0.001

bound_top = np.array([block_all, block_all, block_all, block_all,
                      block_all, bound_all, bound_all, bound_all,
                      bound_all, bound_all, bound_all, block_all,
                      block_all, block_all, block_all, block_all,
                      block_all, block_all,
                      1e20, 1e20])

bound_all = -100.00
block_all = -0.001

bound_bottom = np.array([block_all, block_all, block_all, block_all,
                         block_all, bound_all, bound_all, bound_all,
                         bound_all, bound_all, bound_all, block_all,
                         block_all, block_all, block_all, block_all,
                         block_all, block_all,
                         -1e20, -1e20])
# correction 1

#bound_bottom = np.array([-.145, -.759, -.1, -.553, -.491, -.235,
#                         -.648, -.1, -.489, -.248,
#                         -1e20, -1e20])
#
#bound_top = 1 + bound_bottom
#
#bound_top[-2] = 1e20
#bound_top[-1] = 1e20


if 'Height' in what2do:
    res = lsq_linear(m_matrix, target, bounds=(bound_bottom, bound_top),
                     method='bvls', tol=1e-32, verbose=1, max_iter=1000)
elif 'DPC' in what2do:
    res = lsq_linear(m_matrix, dpc_target, bounds=(bound_bottom, bound_top), verbose=1)
elif 'Curvature' in what2do:
    res = lsq_linear(m_matrix, curv_target, bounds=(bound_bottom, bound_top), verbose=1)

# %%
print('Status: {}'.format(res.status))
if res.success:
    print("Uha!")
else:
    print("FAIL!!!!")
voltage = res.x[:-2]
piston = res.x[-2]
tilt = res.x[-1]

# %%


for i, fname in enumerate(listOfFiles):
    label = fname.rsplit('/', 1)[1].split('.')[0]
    wpu.print_blue(label + '\t: {:6.5f} Volts'.format(voltage[i]))

wpu.print_blue('piston: {:.4g} nm'.format(piston*1e9))
wpu.print_blue('tilt: {:.4g} rad?'.format(tilt))



# %%

# TODO:


#for volt in voltage:
#    print('{:.1f}'.format(volt + base_voltage), end=" ")

#print('')


for nn in range(18):
    print('{:.1f}'.format(voltage[nn] + base_voltage[nn]), end=" ")
    
print('')

for nn in range(18):
    print('{:.1f}'.format(voltage[nn] + base_voltage[nn]), end=", ")
    
print('')
# %%

fname = wpu.get_unique_filename(dirName + '/resp_func_m_matrix_' + what2do.replace(' ', '_'), 'dat')
wpu.save_csv_file(m_matrix,
                  fname=fname)

fname = wpu.get_unique_filename(dirName + '/resp_func_voltage_' + what2do.replace(' ', '_'), 'dat')
wpu.save_csv_file([np.arange(1, np.size(voltage) + 1), voltage],
                  fname=fname,
                  headerList=['Channel', 'Voltage [V]'])

fname = wpu.get_unique_filename(dirName + '/resp_func_target_' + what2do.replace(' ', '_'), 'dat')
wpu.save_csv_file(target,
                  fname=fname,
                  headerList=['Height [m]'])

# %%
voltage4plot = np.zeros(np.size(voltage)+2)
voltage4plot[:-2] = voltage

voltage4plot[-2] = piston
voltage4plot[-1] = tilt

# %%

finalSurface = m_matrix @ voltage4plot
plt.figure()


if 'Height' in what2do:
    plt.plot(xnew*1e6, finalSurface*1e9)
    plt.ylabel('Height [nm]')
else:
    plt.plot(xnew*1e6, finalSurface)
    plt.ylabel(headers[-1])

#plt.title('Final Surface, Correction: {:} V'.format(bound_top))
plt.title('Surface Displacement, rms = {:.2f} pm'.format(np.std(finalSurface)*1e12))

plt.xlabel(r'y [um]')
figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show(block=False)

plt.figure()

if 'Height' in what2do:
    plt.plot(xnew*1e6, -(res.fun-np.mean(res.fun))*1e9, '-', label='Residual')
    plt.plot(xnew*1e6, (target-np.mean(target))*1e9, '-', label='Target')
    plt.ylabel('Height [nm]')
else:
    plt.plot(xnew*1e6, res.fun, label='Residual')
    plt.ylabel(headers[-1])


#plt.title('Residual, Correction: {:} V'.format(bound_top))

plt.title('Target, ' + targetName[0].rsplit('/', 1)[1] +
          ', rms = {:.2f} pm'.format(np.std(target)*1e12) +
          '\nResidual, rms = {:.2f} pm'.format(np.std(res.fun)*1e12))
plt.xlabel(r'y [um]')
plt.legend()
figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.bar(range(1, np.size(voltage) + 1),voltage, width=1.0)
plt.xlabel('Channel #')
plt.ylabel('Voltage [V]')

#plt.title('Final Voltage, Max Correction: {:} V'.format(bound_top))
plt.title('Final Voltage')
figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
plt.savefig(figname)
plt.show()

# %%
for_csv = [xnew]
for_csv.append(finalSurface)
for_csv.append(res.fun)

fname = wpu.get_unique_filename(dirName + '/final_shape_' + what2do.replace(' ', '_'), 'dat')
wpu.save_csv_file(for_csv,
                  fname=fname,
                  headerList=['x[m], Height [m], Residual [m]'])
# %%
#
#test_volt = np.array([-100., -100., 100., -100.,
#                      100., 100., 100., 100.,
#                      -100., -100., -100., -100.,
#                      100., 100., 100., -100.,
#                      -100., -100.])
#
#
#finalSurface_test = m_matrix @ test_volt
#
#
#plt.figure()
#plt.plot(xnew*1e6, finalSurface_test*1e9)
#
#
#plt.ylabel('Height [nm]')
#plt.title('Final Surface')
#plt.xlabel(r'y [um]')
#plt.show(block=False)
#
## %%
#plt.figure()
#
#residual_test = finalSurface_test -target
#residual_test -= np.mean(residual_test)
#
#plt.plot(xnew*1e6, residual_test*1e9)
#
#plt.ylabel('Height [nm]')
#
#
#plt.title('Residual')
#plt.xlabel(r'y [um]')
#plt.show(block=False)

# %%
#
if False:
    plt.figure()
#
##myModel = .145*m_matrix[:, 0] + \
##          .759*m_matrix[:, 1] + \
##          .1*m_matrix[:, 2] + \
##          .553*m_matrix[:, 3] + \
##          .491*m_matrix[:, 4] + \
##          .235*m_matrix[:, 5] + \
##          .648*m_matrix[:, 6] + \
##          .1*m_matrix[:, 7] + \
##          .489*m_matrix[:, 8] + \
##          .248*m_matrix[:, 9]
#
#

    myVoltages = [0., 0., 0., 0.,
                  0., 0., 0., 0.,
                  0., 0., 0., 0.,
                  0., 0., 0., 0.,
                  0., 0., 0.]

#voltage4plot[-2] = 0
#voltage4plot[-1] = 0
    myModel = m_matrix @ voltage4plot


    plt.plot(xnew*1e6, (myModel - np.min((myModel)))*1e9, '-')


#plt.title('Residual, Correction: {:} V'.format(bound_top))

    plt.title('My Model')
    plt.xlabel(r'y [um]')
    figname = wpu.get_unique_filename(dirName + '/respons_func', 'png')
    plt.savefig(figname)
    plt.show(block=False)
#
## %%
#
#plt.figure()
#
#foo = m_matrix[:, 0]
#for i in range(13):
#    print(i)
#    foo = m_matrix[:, i]
#
#    plt.plot(foo, label='{}'.format(i))
#
#plt.legend()
#plt.show()
