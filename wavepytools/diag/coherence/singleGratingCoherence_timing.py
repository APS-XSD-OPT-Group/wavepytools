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


This Script adapts singleGratingCoherence_z_scan.py for timing meassurements.
Try to keep this codes as similar as possible

'''

import dxchange
import numpy as np


plotFourierImages = False

if plotFourierImages:

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


originalDir = os.getcwd()

try:
    os.chdir('/data/share/BeamtimeData/')
except FileNotFoundError:
    try:
        os.chdir('~/workspace/pythonWorkspace/data')
    except:
        pass


samplefileName = easyqt.get_file_names("Choose one of the scan files")[0]


data_dir = samplefileName.rsplit('/', 1)[0]
os.chdir(data_dir)


wpu.print_blue('MESSAGE: Loading files ' +
               samplefileName.rsplit('_', 2)[0] + '*.tif')


expTimeTable = np.loadtxt(easyqt.get_file_names("File with the exposition time table")[0])

#nFilesTable = np.loadtxt(easyqt.get_file_names("File with the number of files per exposition time")[0])
#
#expTimeTable = np.loadtxt('/home/grizolli/DATA/201704_32ID_2/20170408/timescan/timeTable.txt')
#
#nFilesTable = np.loadtxt('/home/grizolli/DATA/201704_32ID_2/20170408/timescan/nImages.txt')
#
#expTimeTable = np.linspace(0, 9, 40)
nFilesTable = expTimeTable*0.0 + 1.0

#image_per_point = easyqt.get_int('Number of images by point', default_value=1)

timeVec = np.array([])
for i in range(expTimeTable.size):

    timeVec = np.concatenate((timeVec,
                              expTimeTable[i].repeat(nFilesTable[i])))


#fname_dark =  easyqt.get_file_names("Dark File")[0]

os.chdir(originalDir)



detectorDistance = easyqt.get_float('Detector distance [mm]',
                                    default_value=8)*1e-3


avgDark = easyqt.get_float('Average Dark  [Counts]',
                                      default_value=100)


listOfDataFiles = glob.glob(samplefileName.rsplit('_', 2)[0] + '*.tif')
listOfDataFiles.sort()
nfiles = len(listOfDataFiles)

if nfiles != np.sum(nFilesTable):

    print('ERROR: There is an error with the number of files! EXIT!!')
    print('ERROR: number of files: {}'.format(nfiles))
    print('ERROR: number of expected files : {}'.format(np.sum(nFilesTable)))
    exit()
# %%
#dark_im = dxchange.read_tiff(fname_dark)

dark_im = dxchange.read_tiff(listOfDataFiles[0])*0.0 + avgDark

img = dxchange.read_tiff(listOfDataFiles[0]) - dark_im

# =============================================================================
# %% Experimental parameters
# =============================================================================

pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                             title='Experimental Values',
                             default_value=.65)*1e-6

gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                 title='Experimental Values',
                                 default_value=2.3)*1e-6


pattern = easyqt.get_choice(message='Select CB Grating Pattern', title='Title',
                            choices=['Diagonal', 'Edge'])

sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                  title='Experimental Values', max_=99999,
                                  default_value=39.0)


# =============================================================================
# %% Crop
# =============================================================================

#idx = [200, -200, 0, 2160]

idx = [0, -1, 0, -1]

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

plt.figure()
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
 _) = wgi.exp_harm_period(img, [period_harm_Vert,
                                         period_harm_Horz],
                                         harmonic_ij=['1', '0'],
                                         searchRegion=40,
                                         isFFT=False, verbose=True)

(_,
 period_harm_Horz) = wgi.exp_harm_period(img, [period_harm_Vert,
                                         period_harm_Horz],
                                         harmonic_ij=['0', '1'],
                                         searchRegion=40,
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

    wpu.print_blue("MESSAGE: loop " + str(i) + ": " +
                   listOfDataFiles[i])

    if idx == [0, -1, 0, -1]:
        img = dxchange.read_tiff(listOfDataFiles[i]) - dark_im
    else:
        img = wpu.crop_matrix_at_indexes(dxchange.read_tiff(listOfDataFiles[i]), idx) - dark_im

    #    pv, ph = period_harm_Vert, period_harm_Horz

    pv = int(period_harm_Vert /
             (sourceDistance + detectorDistance)*(sourceDistance))
    ph = int(period_harm_Horz /
             (sourceDistance + detectorDistance)*(sourceDistance))


    if plotFourierImages:

        wgi.plot_harmonic_grid(img,
                               [pv, ph],
                               isFFT=False)

        plt.savefig('FFT_{:.0f}ms.png'.format(timeVec[i]*1e3))
        plt.show(block=False)
        plt.close()

        wgi.plot_harmonic_peak(img,
                               [pv, ph],
                               isFFT=False)

        plt.savefig('FFT_peaks_{:.0f}ms.png'.format(timeVec[i]*1e3))
        plt.show(block=False)
        plt.close()

    return wgi.visib_1st_harmonics(img, [pv, ph], searchRegion=40)


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

contrastV = np.asarray(res)[:, 0]
contrastH = np.asarray(res)[:, 1]


# =============================================================================
# %% Plot
# =============================================================================


#fig = plt.figure()
#plt.plot(timeVec, contrastV*100, '-ko', label='Vert')
#plt.plot(timeVec, contrastH*100, '-ro', label='Hor')
#plt.xlabel(r'Time [s]', fontsize=14)
#
#
#plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
#plt.title('Visibility vs detector distance', fontsize=14, weight='bold')
#plt.grid()
#plt.legend(fontsize=14)
#
#wpu.save_figs_with_idx_pickle(fig, samplefileName.rsplit('_' ,1)[0].rsplit('/' ,1)[1])
#plt.show(block=True)




# =============================================================================
# %% Plot
# =============================================================================

uniqueTimeVec = np.unique(timeVec)
meanV = uniqueTimeVec*0.0
stdV = uniqueTimeVec*0.0
maxCv = uniqueTimeVec*0.0
minCv = uniqueTimeVec*0.0

for i in range(uniqueTimeVec.size):

    meanV[i] = np.mean(contrastV[np.argwhere(timeVec==uniqueTimeVec[i])])
    maxCv[i] = np.max(contrastV[np.argwhere(timeVec==uniqueTimeVec[i])])
    minCv[i] = np.min(contrastV[np.argwhere(timeVec==uniqueTimeVec[i])])

    stdV[i] = np.std(contrastV[np.argwhere(timeVec==uniqueTimeVec[i])])

fig = plt.figure(figsize=(12,8))
plt.semilogx(timeVec, contrastV*100, '-ko', label='Vert')
plt.semilogx(timeVec, contrastH*100, '-ro', label='Hor')

plt.semilogx(uniqueTimeVec, meanV*100, '-b', lw=3, label='Mean')
plt.semilogx(uniqueTimeVec, (stdV)*100, '-c', lw=3, label='Std')
#plt.semilogx(uniqueTimeVec, (meanV-stdV)*100, '--c', lw=3, label='Std')
plt.semilogx(uniqueTimeVec, maxCv*100, '--g', lw=3, label='Max')
plt.semilogx(uniqueTimeVec, minCv*100, '--g', lw=3, label='Min')

plt.xlabel(r'Time [s]', fontsize=14)


plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
plt.title('Visibility vs Exposition time,' +
           ' d={:.0f}mm'.format(detectorDistance*1e3), fontsize=14, weight='bold')
plt.grid()
plt.legend(fontsize=14)

wpu.save_figs_with_idx_pickle(fig, 'z{:.0f}mm_'.format(detectorDistance*1e3) +
                              samplefileName.rsplit('_' ,1)[0].rsplit('/' ,1)[1])

wpu.save_figs_with_idx('z{:.0f}mm_'.format(detectorDistance*1e3) +
                              samplefileName.rsplit('_' ,1)[0].rsplit('/' ,1)[1])
plt.show(block=True)
