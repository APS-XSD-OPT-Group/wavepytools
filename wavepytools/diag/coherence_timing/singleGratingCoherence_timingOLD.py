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

#import matplotlib
#matplotlib.use('Agg')


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


originalDir = os.getcwd()

try:
    os.chdir('/data/share/BeamtimeData/')
except FileNotFoundError:
    try:
        os.chdir('~/workspace/pythonWorkspace/data')
    except:
        pass



samplefileName =  easyqt.get_file_names("Choose one of the scan files")[0]
listOfDataFiles = glob.glob(samplefileName.rsplit('_' ,1)[0] + '*.tif')

data_dir = samplefileName.rsplit('/', 1)[0]
os.chdir(data_dir)

listOfDataFiles.sort()

wpu.print_blue('MESSAGE: Loading files ' + \
                samplefileName.rsplit('_',1)[0] + '*.tif')

#fname_dark =  easyqt.get_file_names("Dark File")[0]

os.chdir(originalDir)


strideFile = easyqt.get_int('Stride', default_value=1)


startDist = easyqt.get_float('Starting distance scan [mm]',
                           default_value=10)*1e-3

step_z_scan = strideFile*easyqt.get_float('Step size scan [mm]',
                                        default_value=2)*1e-3


listOfDataFiles = listOfDataFiles[0::strideFile]
nfiles = len(listOfDataFiles)


#dark_im = dxchange.read_tiff(fname_dark)

dark_im = dxchange.read_tiff(listOfDataFiles[0])*0.0
img = dxchange.read_tiff(listOfDataFiles[0]) - dark_im

# =============================================================================
# %% Experimental parameters
# =============================================================================

pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                             title='Experimental Values',
                             default_value=.65)*1e-6

gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                 title='Experimental Values',
                                 default_value=4.8)*1e-6


pattern = easyqt.get_choice(message='Select CB Grating Pattern', title='Title',
                             choices=['Edge', 'Diagonal'])

sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                  title='Experimental Values',
                                  default_value=62)


zvec = np.linspace(startDist, startDist + step_z_scan*(nfiles-1), nfiles)

zvec = np.array([0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.017,
                 0.019, 0.021, 0.023, 0.025, 0.028, 0.030, 0.033, 0.037, 0.040,
                 0.044, 0.049, 0.054, 0.059, 0.065, 0.072, 0.079, 0.087, 0.095,
                 0.105, 0.115, 0.127, 0.140, 0.154, 0.169, 0.186, 0.204, 0.225,
                 0.247, 0.272, 0.299, 0.329, 0.362, 0.398, 0.438, 0.482, 0.530,
                 0.583, 0.641, 0.706, 0.776, 0.854, 0.939, 1.033, 1.136, 1.250,
                 1.375, 1.512, 1.664, 1.830, 2.013])



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

if pattern == 'Diagonal':
    period_harm_Vert = np.int(np.sqrt(2)*pixelSize/gratingPeriod*img.shape[0])
    period_harm_Horz = np.int(np.sqrt(2)*pixelSize/gratingPeriod*img.shape[1])

elif pattern == 'Edge':
    period_harm_Vert = np.int(2*pixelSize/gratingPeriod*img.shape[0])
    period_harm_Horz = np.int(2*pixelSize/gratingPeriod*img.shape[1])

# Obtain harmonic periods from images

(period_harm_Vert,
 period_harm_Horz) = wgi.exp_harm_period(img, [period_harm_Vert,
                                        period_harm_Horz],
                                        harmonic_ij=['1', '1'],
                                        searchRegion=20,
                                        isFFT=False, verbose=True)





# =============================================================================
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


fig = plt.figure()
plt.plot(zvec*1e3, contrastV*100, '-ko', label='Vert')
plt.plot(zvec*1e3, contrastH*100, '-ro', label='Hor')
plt.xlabel(r'Distance $z$  [mm]', fontsize=14)

#plt.plot(range(nfiles), contrastV*100, '-ro', label='Vert')
#plt.plot(range(nfiles), contrastH*100, '-go', label='Hor')
#plt.xlabel(r'Image number', fontsize=14)

plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
plt.title('Visibility vs detector distance', fontsize=14, weight='bold')
plt.grid()
plt.legend(fontsize=14)
plt.show()


wpu.save_figs_with_idx_pickle(fig, samplefileName.rsplit('_' ,1)[0].rsplit('/' ,1)[1])






