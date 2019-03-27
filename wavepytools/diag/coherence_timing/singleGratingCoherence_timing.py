#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

'''
Author: Walan Grizolli


This Script use the technique described by Xianbo Shi in
https://doi.org/10.1364/OE.22.014041

'''

import dxchange
import numpy as np

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

import glob

import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi
import wavepy.surface_from_grad as wps


import itertools
from multiprocessing import Pool, cpu_count
import time

import os
from wavepy.utils import easyqt

# =============================================================================
# %% Load Image
# =============================================================================




data_dir =  easyqt.get_directory_name("Choose directory with all files")


os.chdir(data_dir)

listOfDataFiles = glob.glob(data_dir+ '/**.tif')


listOfDataFiles.sort()

#wpu.print_blue('MESSAGE: Loading files ' + \
#                samplefileName.rsplit('_',1)[0] + '*.tif')

#fname_dark =  easyqt.get_file_names("Dark File")[0]


nfiles = len(listOfDataFiles)


#dark_im = dxchange.read_tiff(fname_dark)

dark_im = dxchange.read_tiff(listOfDataFiles[0])*0.0
img = dxchange.read_tiff(listOfDataFiles[0]) - dark_im

# %%
#exit()

# =============================================================================
# %% Experimental parameters
# =============================================================================


from scipy import constants
hc = constants.value('inverse meter-electron volt relationship')  # hc

wavelength = hc/easyqt.get_float("Photon Energy [KeV]",
                                     title='Experimental Values',
                                     default_value=25.0)*1e-3


pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                             title='Experimental Values',
                             default_value=.65)*1e-6

gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                 title='Experimental Values',
                                 default_value=3.4)*1e-6

pattern = easyqt.get_choice(message='Select CB Grating Pattern', title='Title',
                             choices=['Diagonal', 'Edge'])

if pattern == 'Diagonal':
    gratingPeriod *= 1.0/np.sqrt(2.0)

elif pattern == 'Edge':
    gratingPeriod *= 1.0/2.0

sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                  title='Experimental Values',
                                  default_value=40)



tvec_labels = np.array([0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.017,
                        0.019, 0.021, 0.023, 0.025, 0.028, 0.030, 0.033, 0.037, 0.040,
                        0.044, 0.049, 0.054, 0.059, 0.065, 0.072, 0.079, 0.087, 0.095,
                        0.105, 0.115, 0.127, 0.140, 0.154, 0.169, 0.186, 0.204, 0.225,
                        0.247, 0.272, 0.299, 0.329, 0.362, 0.398, 0.438, 0.482, 0.530,
                        0.583, 0.641, 0.706, 0.776, 0.854, 0.939, 1.033, 1.136, 1.250,
                        1.375, 1.512, 1.664, 1.830, 2.013])
#
#tvec_labels = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
#                        0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
#                        0.017, 0.018, 0.019])

zvec = np.zeros(nfiles)
tvec = np.zeros(nfiles)

for i in range(nfiles):

    zvec[i] = float(listOfDataFiles[i].rsplit('mm')[0].split('_')[-1])*1e-3
    tvec[i] = tvec_labels[int(listOfDataFiles[i].rsplit('.')[0].split('_')[-1]) - 1]


# =============================================================================
# %% Crop
# =============================================================================

idx = [200, -200, 0, 2160]

img = wpu.crop_matrix_at_indexes(img, idx)

wpu.print_blue("MESSAGE: idx for cropping")
wpu.print_blue(idx)


if idx != [0, -1, 0, -1]:
    dark_im = wpu.crop_matrix_at_indexes(dark_im, idx)

    idx[1] = idx[0] + np.max([idx[1]-idx[0], idx[3] - idx[2]])

    idx[3] = idx[2] + np.max([idx[1]-idx[0], idx[3] - idx[2]])


# =============================================================================
# %% Plot Real Image
# =============================================================================
#
plt.imshow(img, cmap='Greys')
plt.colorbar()
plt.title('Raw Image', fontsize=18, weight='bold')
plt.show(block=True)


# ==============================================================================
# %% Harmonic Periods
# ==============================================================================

period_harm_Vert = np.int(pixelSize/gratingPeriod*img.shape[0])
period_harm_Horz = np.int(pixelSize/gratingPeriod*img.shape[1])


# Obtain harmonic periods from images

(period_harm_Vert,
 period_harm_Horz) = wgi.exp_harm_period(img, [period_harm_Vert,
                                        period_harm_Horz],
                                        harmonic_ij=['1', '1'],
                                        searchRegion=20,
                                        isFFT=False, verbose=True)


## =============================================================================
# %% Calculate everything
# =============================================================================

# =============================================================================
# %% Function for multiprocessing
# =============================================================================

_mapCounter = itertools.count()
next(_mapCounter)



def _func(i):

    wpu.print_blue("MESSAGE: loop " + str(i) + ": " + \
                   listOfDataFiles[i])

    if idx == [0, -1, 0, -1]:
        img = dxchange.read_tiff(listOfDataFiles[i]) - dark_im
    else:
        img = wpu.crop_matrix_at_indexes(dxchange.read_tiff(listOfDataFiles[i]), idx) - dark_im


    pv, ph = period_harm_Vert, period_harm_Horz

    pv = int(period_harm_Vert/(sourceDistance + zvec[i]-zvec[0])*sourceDistance)
    ph = int(period_harm_Horz/(sourceDistance + zvec[i]-zvec[0])*sourceDistance)

    wgi.plot_harmonic_grid(img,
                           [pv, ph],
                           isFFT=False)

    plt.savefig('FFT_{:04.0f}_ms_'.format(tvec[i]*1e3) +
                '{:04.0f}mm.png'.format(zvec[i]*1e3))
    plt.show(block=False)
    plt.close()

#    wgi.plot_harmonic_peak(img,
#                           [pv, ph],
#                           isFFT=False,
#                           fname='FFT_peaks_{:04.0f}_ms_'.format(tvec[i]*1e3) +
#                                   '{:04.0f}mm.png'.format(zvec[i]*1e3))
#
#
#    plt.close()


    return wgi.visib_1st_harmonics(img, [pv, ph], searchRegion=10)


# =============================================================================
# %% multiprocessing
# =============================================================================

ncpus = cpu_count()
wpu.print_blue("MESSAGE: %d cpu's available" % ncpus)


tzero = time.time()

p = Pool(6)
res = p.map(_func, range(nfiles))
p.close()


wpu.print_blue('MESSAGE: Time spent: {0:.3f} s'.format(time.time() - tzero))


# =============================================================================
# %% Sorting the data
# =============================================================================

contrastV = np.asarray(res)[:,0]
contrastH = np.asarray(res)[:,1]





# =============================================================================
# %% Plot
# =============================================================================

tvalues = list(set(tvec))
tvalues.sort()


from scipy.optimize import curve_fit

def _func_4_fit(z, Amp, sigma):
    return Amp*np.exp(-z**2/2/sigma**2)

p0 = [1.0, 0.1]
#bounds = ([1e-3, 0.1, .01, -1., .001],
#          [2.0,   1.0, 100 , 1., .1])

# %%

cohLength_h = []
cohLength_v = []


for acqTime in tvalues:

    fig = plt.figure()

    # Vert
    plt.plot(zvec[np.where(tvec == acqTime)]*1e3,
             contrastV[np.where(tvec == acqTime)]*100, '-ko', label='Vert')

    poptV, pcovV = curve_fit(_func_4_fit,
                           zvec[np.where(tvec == acqTime)],
                           contrastV[np.where(tvec == acqTime)], p0=p0)

    plt.plot(np.linspace(np.min(zvec), np.max(zvec), 101)*1e3,
             _func_4_fit(np.linspace(np.min(zvec), np.max(zvec), 101),
                         poptV[0], poptV[1])*100,
             '--g', label='Fit V')

    cohLength_v.append(np.abs(poptV[1])*wavelength/(gratingPeriod))


    # Hor
    plt.plot(zvec[np.where(tvec == acqTime)]*1e3,
             contrastH[np.where(tvec == acqTime)]*100, '-ro', label='Hor')

    poptH, pcovH = curve_fit(_func_4_fit,
                            zvec[np.where(tvec == acqTime)],
                            contrastH[np.where(tvec == acqTime)], p0=p0)

    plt.plot(np.linspace(np.min(zvec), np.max(zvec), 101)*1e3,
             _func_4_fit(np.linspace(np.min(zvec), np.max(zvec), 101),
                         poptH[0], poptH[1])*100,
             '--b', label='Fit H')

    cohLength_h.append(np.abs(poptH[1])*wavelength/(gratingPeriod))



    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
    plt.xlabel(r'Distance  [mm]', fontsize=14)
    plt.title(r'$\Delta t$ = {}ms'.format(acqTime*1e3) +
              r', coh length V: {:.2f} um'.format(cohLength_v[-1]*1e6) +
              r', coh length H: {:.2f} um'.format(cohLength_h[-1]*1e6))

    plt.legend()

    plt.savefig('visb_t_{:.0f}ms.png'.format(acqTime*1e3))

# %%

cohLength_h = np.asarray(cohLength_h)
cohLength_v = np.asarray(cohLength_v)


fig = plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(tvalues, cohLength_v*1e6, '-ko', label='Vert')
#plt.semilogx(tvalues, cohLength_v*1e6, '-ko', label='Vert')
plt.ylabel(r'Coherence Length [$\mu m$]', fontsize=14)
plt.xlabel(r'$\Delta t$ [mm]', fontsize=14)
plt.grid()
plt.legend()


plt.subplot(122)
plt.plot(tvalues, cohLength_h*1e6, '-ro', label='Hor')
#plt.semilogx(tvalues, cohLength_h*1e6, '-ro', label='Hor')
plt.ylabel(r'Coherence Length [$\mu m$]', fontsize=14)
plt.xlabel(r'$\Delta t$ [s]', fontsize=14)
plt.grid()
plt.legend()

plt.savefig('coh_length.png'.format(acqTime*1e3))


wpu.save_figs_with_idx_pickle(fig)

np.savetxt('coh_length.csv',
           np.array((tvalues, cohLength_v*1e6, cohLength_h*1e6)).T,
           fmt='%1.5g',
           delimiter=',',
           header='Time[s], Coh Length Vert [um], Coh Length Hor [um]')

# =============================================================================
# %% Plot
# =============================================================================

zvalues = list(set(zvec))
zvalues.sort()

for dist in zvalues:


    fig = plt.figure()

    plt.plot(tvec[np.where(zvec == dist)]*1e3,
             contrastV[np.where(zvec == dist)]*100, '-ko', label='Vert')

    plt.plot(tvec[np.where(zvec == dist)]*1e3,
             contrastH[np.where(zvec == dist)]*100, '-ro', label='Hor')

    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
    plt.xlabel(r'time  [ms]', fontsize=14)
    plt.title(r'$z$ = {:.0f}mm'.format(dist*1e3))

    plt.savefig('visb_d_{:.0f}mm.png'.format(dist*1e3))



