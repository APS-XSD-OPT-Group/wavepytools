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

used to load and fit the results from singleGratingCoherence_z_scan.py when we
have only the peaks

'''


import pickle
from wavepy.utils import easyqt
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import argrelmin, argrelmax

from scipy import constants

import wavepy.utils as wpu

hc = constants.value('inverse meter-electron volt relationship')  # hc


# %%
def _chi2(data, model):
    eps = np.spacing(1)
    data_zerofree = np.where(np.abs(data) > eps, data, eps)
    return np.sum(np.abs(data_zerofree-model)/np.max(data_zerofree))/data.size


# %%
def max_min_mean_4plot(vec_x, vec_y):

    unique_x = np.unique(vec_x)

    max_y = unique_x*0.0
    min_y = unique_x*0.0
    mean_y = unique_x*0.0

    for n in range(unique_x.size):

        max_y[n] = np.max(vec_y[np.argwhere(vec_x == unique_x[n])])
        min_y[n] = np.min(vec_y[np.argwhere(vec_x == unique_x[n])])
        mean_y[n] = np.mean(vec_y[np.argwhere(vec_x == unique_x[n])])

    return max_y, min_y, mean_y


# %%
def fit_peaks_talbot(zvec, contrast, wavelength,
                     patternPeriod, sourceDist,
                     fname4graphs, title4graph='Title'):

    def _func_4_fit(z, Amp, sigma):
        return (Amp*np.exp(-z**2/2/sigma**2))

    p0 = [1.0, np.sqrt(np.sum(contrast*zvec**2)/np.sum(contrastV))]
    # bounds = ([1e-3, 0.1, .01, -1., .001],
    #          [2.0,   1.0, 100 , 1., .1])

    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0)

    print("Fit 1D")
    print("Amp, sigma")
    print(popt)

    cohLength = np.abs(popt[1])*wavelength/patternPeriod

    print('Coherent length: {:.4g} um'.format(cohLength*1e6))

    plt.figure(figsize=(12, 9))
    plt.plot(zvec*1e3, 100*contrast, 'ok', ms=7, label='data')

    fitted_values = _func_4_fit(zvec, popt[0], popt[1])
    chi2 = _chi2(contrast, fitted_values)
    wpu.print_blue('chi2 Fit = {:.3f}'.format(chi2))

    zvec_4_fit = np.linspace(zvec[0], zvec[-1], zvec.size*5)
    plt.plot(zvec_4_fit*1e3,
             100*_func_4_fit(zvec_4_fit, popt[0], popt[1]),
             '-r', lw=3, label='Fit')

    max_cont, min_cont, mean_cont = max_min_mean_4plot(zvec, contrast)
    plt.plot(np.unique(zvec)*1e3, 100*max_cont, '--c', lw=2, label='Max')
    plt.plot(np.unique(zvec)*1e3, 100*min_cont, '--c', lw=2, label='Min')
    plt.plot(np.unique(zvec)*1e3, 100*mean_cont, '--m', lw=2, label='Mean')

    title4graph += r', $l_{coh}$ =' + ' {:.3f} um'.format(cohLength*1e6)
    title4graph += r', $\chi^2$ = {:.3f}'.format(chi2)

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph, fontsize=27, weight='bold')
    plt.legend(fontsize=22)
    plt.grid()
    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=True)

    return cohLength, popt  # zperiod


# %%


def plot_max_min_mean(zvec, contrast, fname4graphs, title4graph='Title'):

    plt.figure(figsize=(12, 9))
    plt.plot(zvec*1e3, 100*contrast, 'ok', ms=7, label='data')

    max_cont, min_cont, mean_cont = max_min_mean_4plot(zvec, contrast)
    plt.plot(np.unique(zvec)*1e3, 100*max_cont, '--c', lw=2, label='Max')
    plt.plot(np.unique(zvec)*1e3, 100*min_cont, '--c', lw=2, label='Min')
    plt.plot(np.unique(zvec)*1e3, 100*mean_cont, '--m', lw=2, label='Mean')

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph, fontsize=27, weight='bold')
    plt.legend(fontsize=22)
    plt.grid()
    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=True)




# %%
def fit_peaks_talbot2gauss(zvec, contrast, wavelength,
                           patternPeriod, sourceDist,
                           cohLength,
                           fname4graphs, title4graph='Title'):

    def _func_4_fit(z, Amp, Amp2, sigma1, sigma2):
        return Amp*np.exp(-z**2/2/sigma1**2) + Amp2*np.exp(-z**2/2/sigma2**2)

    cohL_o = cohLength/wavelength*patternPeriod
    p0 = [1.0, .5, cohL_o, cohL_o*.15]
    bounds = ([.01, .01, cohL_o*0.5, cohL_o*0.01],
              [2.0,  2.0, cohL_o*1.5, cohL_o*1.0])

    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0,
                           bounds=bounds)

    print("Fit 1D")
    print("Amp, z_period, sigma, phase")
    print(popt)

    cohLength1 = np.abs(popt[2])*wavelength/patternPeriod
    cohLength2 = np.abs(popt[3])*wavelength/patternPeriod

    print('Coherent length: {:.4g} um'.format(cohLength1*1e6))
    print('Coherent length: {:.4g} um'.format(cohLength2*1e6))

    plt.figure(figsize=(12, 9))
    plt.plot(zvec*1e3, 100*contrast, 'ok', ms=7, label='data')

    fitted_values = _func_4_fit(zvec, popt[0], popt[1],
                                popt[2], popt[3])
    chi2 = _chi2(contrast, fitted_values)
    wpu.print_blue('chi2 Fit = {:.3f}'.format(chi2))

    zvec_4_fit = np.linspace(zvec[0], zvec[-1], zvec.size*5)
    plt.plot(zvec_4_fit*1e3,
             100*_func_4_fit(zvec_4_fit, popt[0], popt[1], popt[2], popt[3]),
             '-r', lw=3, label='Fit')

    max_cont, min_cont, mean_cont = max_min_mean_4plot(zvec, contrast)
    plt.plot(np.unique(zvec)*1e3, 100*max_cont, '--c', lw=2, label='Max')
    plt.plot(np.unique(zvec)*1e3, 100*min_cont, '--c', lw=2, label='Min')
    plt.plot(np.unique(zvec)*1e3, 100*mean_cont, '--m', lw=2, label='Mean')

    title4graph += r', $l_{coh}$ =' + ' {:.3f} um'.format(cohLength1*1e6)
    title4graph += r', $l_{coh}$ =' + ' {:.3f} um'.format(cohLength2*1e6)
    title4graph += r', $\chi^2$ = {:.3f}'.format(chi2)

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph, fontsize=27, weight='bold')
    plt.legend(fontsize=22)
    plt.grid()
    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=True)

    return cohLength1, cohLength2, popt  # zperiod


# %%
def _load_data_from_pickle(fname):

    fig = pickle.load(open(fname, 'rb'))
    fig.set_size_inches((12, 9), forward=True)

    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=True)  # this lines keep the script alive to see the plot

    curves = []

    for i in range(len(fig.axes[0].lines)):

        curves.append(np.asarray(fig.axes[0].lines[i].get_data()))

    return curves


# %% Coh Function fit Bessel

def _coh_func_fit_bessel(coh_function, coh_func_coord,
                         wavelength, sourceDistance,
                         title4graph='Title', saveGraphs=False):
    '''
    TO BE FINISHED!!!!
    '''

    coh_function *= 1/np.max(coh_function)

    plt.figure(figsize=(12, 9))
    plt.plot(coh_func_coord*1e6, coh_function, '-og', lw=3,
             label='DOC function - experimental envelop')
    plt.xlabel(title4graph + r' Position [$\mu m$]', fontsize=27)
    plt.ylabel('Coh Function', fontsize=27)
    plt.title(title4graph + ' Coh Function', fontsize=27, weight='bold')

    from scipy.ndimage.filters import gaussian_filter
    zeros_arg = argrelmin(gaussian_filter(coh_function, 5), order=10)

    print('zeros')
    print(coh_func_coord[zeros_arg]*1e6)

    zero1 = np.min(np.abs(coh_func_coord[zeros_arg]))

    plt.plot(coh_func_coord[zeros_arg]*1e6,
             coh_function[zeros_arg], 'sr',
             label='Minima')

    from scipy.special import j0

    def _func4fitCoh(x, p0, p1, p2):

        sigma = p2/2.35

        return p0 * np.abs(j0(x * p1)) * np.exp(-x**2/2/sigma**2)

    p0 = [10.000, 2.40482556/zero1, 50e-6]

    arg4fit = np.where(np.abs(coh_func_coord) > 1e-6)

    popt, pcov = curve_fit(_func4fitCoh,
                           coh_func_coord[arg4fit],
                           coh_function[arg4fit], p0=p0)

    yamp = wavelength*sourceDistance*popt[1]/2/np.pi
    beam_size = wavelength*sourceDistance/popt[2]

    print("Fit bessel")
    print("Amp, y_o, FWHM beam")
    print('{:.3f} {:.3f}um {:.3f}um'.format(popt[0], yamp*1e6, beam_size*1e6))

    fitted_func = _func4fitCoh(coh_func_coord, popt[0], popt[1], popt[2])
    gauss_envelope = _func4fitCoh(coh_func_coord, popt[0], 0, popt[2])

    plt.plot(coh_func_coord*1e6, fitted_func, '--m',
             lw=3, label='Fitted Function')

    plt.plot(coh_func_coord*1e6, gauss_envelope, '--c',
             lw=3, label='Gaussian Envelop')

    plt.title(title4graph +
              r' Fit, $y_o$:{:.1f}um,'.format(yamp*1e6) +
              r' $\Delta_{source}$' + ':{:.1f}um'.format(popt[2]*1e6),
              fontsize=27, weight='bold')

    plt.legend(loc=1, fontsize=14)
    plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=True)

    return coh_func_coord, fitted_func



# %%

if __name__ == '__main__':
    """
    To run in the command line use:
    fit_singleGratingCoherence_z_peaks.py arg1 arg2 arg3 arg4 arg5



    arg1:    photon Energy [KeV]
    arg2:    Grating Period [µm]
    arg3:    CB Grating Orientation: 'Edge' or 'Diagonal'
    arg4:    Source distance [meters]
    arg5:    file name [.pickle]

    """


    if len(sys.argv) == 1:

        flist = easyqt.get_file_names("Pickle File to Plot")

        fname = flist[0]

        wavelength = hc/easyqt.get_float("Photon Energy [KeV]",
                                         title='Experimental Values',
                                         default_value=8.0)*1e-3

        grPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                    title='Experimental Values',
                                    default_value=4.8)*1e-6

        pattern = easyqt.get_choice(message='Select CB Grating Pattern',
                                    title='Experimental Values',
                                    choices=['Edge', 'Diagonal'])
#                                    choices=['Diagonal', 'Edge'])

        sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                          title='Experimental Values',
                                          default_value=34)

    else:

        wavelength = hc/float(sys.argv[1])*1e-3

        grPeriod = float(sys.argv[2])*1e-6

        pattern = sys.argv[3]

        sourceDistance = float(sys.argv[4])

        fname = sys.argv[5]

    fname4graphs = fname.rsplit('.')[0]
    font = {'family': 'Bitstream Vera Sans',
            'size':   25}

    plt.rc('font', **font)

    results = _load_data_from_pickle(fname)

    zvec = results[0][0]*1e-3
    contrastV = results[0][1]*1e-2
    contrastV -= np.min(contrastV)
    contrastH = results[1][1]*1e-2
    contrastH -= np.min(contrastH)

    if zvec[-1] - zvec[0] < 0:
        zvec = zvec[::-1]
        contrastV = contrastV[::-1]
        contrastH = contrastH[::-1]

    if pattern == 'Diagonal':
        patternPeriod = grPeriod*np.sqrt(2)/2
    elif pattern == 'Edge':
        patternPeriod = grPeriod/2


# %% fit_peaks_talbot Vertical

    plot_max_min_mean(zvec, contrastV,
                      fname4graphs=fname4graphs,
                      title4graph='Vertical')

    cohLength_V, _ = fit_peaks_talbot(zvec, contrastV, wavelength,
                                      patternPeriod, sourceDistance,
                                      fname4graphs=fname4graphs,
                                      title4graph='Vertical')

    beam_sizeV = wavelength*sourceDistance/cohLength_V/2/np.pi
    print('Beam Size Vertical: {:.2f}um'.format(beam_sizeV*1e6))


# %%
    res = fit_peaks_talbot2gauss(zvec, contrastV, wavelength,
                                 patternPeriod, sourceDistance,
                                 cohLength=cohLength_V,
                                 fname4graphs=fname4graphs,
                                 title4graph='Vertical')

    cohLength_V1, cohLength_V2, _ = res

    beam_sizeV1 = wavelength*sourceDistance/cohLength_V1/2/np.pi
    print('Beam Size Vertical1: {:.2f}um'.format(beam_sizeV1*1e6))

    beam_sizeV2 = wavelength*sourceDistance/cohLength_V2/2/np.pi
    print('Beam Size Vertical2: {:.2f}um'.format(beam_sizeV2*1e6))



# %% fit_peaks_talbot Horizontal


    plot_max_min_mean(zvec, contrastH,
                      fname4graphs=fname4graphs,
                      title4graph='Horizontal')

    cohLength_H, _ = fit_peaks_talbot(zvec, contrastH, wavelength,
                                       patternPeriod, sourceDistance,
                                       fname4graphs=fname4graphs,
                                       title4graph='Horizontal')

    beam_sizeH = wavelength*sourceDistance/cohLength_H/2/np.pi
    print('Beam Size Vertical: {:.2f}um'.format(beam_sizeH*1e6))




# %%
    res = fit_peaks_talbot2gauss(zvec, contrastH, wavelength,
                                  patternPeriod, sourceDistance,
                                  cohLength=cohLength_H,
                                  fname4graphs=fname4graphs,
                                  title4graph='Horizontal')

    cohLength_H1, cohLength_H2, _ = res

    beam_sizeV1 = wavelength*sourceDistance/cohLength_H1/2/np.pi
    print('Beam Size Vertical1: {:.2f}um'.format(beam_sizeV1*1e6))

    beam_sizeV2 = wavelength*sourceDistance/cohLength_H2/2/np.pi
    print('Beam Size Vertical2: {:.2f}um'.format(beam_sizeV2*1e6))
