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
#       Laboratory, ANL, the U.S. Government, nor the names of its        #\
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

uses to load and fit the results from singleGratingCoherence_z_scan.py

'''


import pickle
from wavepy.utils import easyqt
import sys

import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import argrelmin, argrelmax

from scipy import constants

import wavepy.utils as wpu

from scipy.ndimage.filters import gaussian_filter, uniform_filter1d


hc = constants.value('inverse meter-electron volt relationship')  # hc

wpu._mpl_settings_4_nice_graphs()

besselzero1 = 2.40482556

import os

def _load_experimental_pars(argv):

    global gui_mode
    global inifname  # name of .ini file
    inifname = os.curdir + '/.' + os.path.basename(__file__).replace('.py', '.ini')

    if len(argv) == 6:

        fname = argv[1]
        phenergy = float(argv[2])*1e3
        a_y = float(argv[3])*1e-6

        menu_options = (int('0b' + argv[4], 2) << 4) + int('0b' + argv[5], 2)

    elif len(argv) == 1:

        try:
            defaults = wpu.load_ini_file(inifname)
            p1 = float(defaults['Parameters'].get('photon energy [kev]'))
            p2 = float(defaults['Parameters'].get('Ay for vibration [um]'))

        except Warning:
            p1, p2 = [0, 0]

        fname = easyqt.get_file_names('csv File to Plot')
        if fname == []:
            fname = defaults['Files'].get('Data')
        else:
            fname = fname[0]

        wpu.print_blue('MESSAGE: Loading file: ' + fname)

        phenergy = easyqt.get_float('photon energy [kev]',
                                    title='Experimental Values',
                                    default_value=p1)*1e3

        a_y = easyqt.get_float('Nominal Vibration Amplitude [um]\n' +
                               '(negative value skip vibration fit)',
                               title='Experimental Values',
                               default_value=p2)*1e-6

        menu_options = 0b101000

    else:

        argExplanations = [' arg0: ',
                           ' arg1:    file name [.csv]',
                           ' arg2:    photon Energy [KeV]',
                           ' arg3:    ay [µm] for vibrations study'
                           ' (use -1 to ignore)',
                           ' arg4:    Other bitwise flags:'
                           ' Two Gaussians Fit, Source calculation',
                           ' arg5:    What to plot, bitwise flags:'
                           ' All Data, Maximum, Minimum, Mean']

        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n'
              'fit_singleGratingCoherence_z_scan.py.py : (no inputs) load dialogs \n'
              '\n'
              'fit_singleGratingCoherence_z_scan.py.py [args] \n'
              '\n')

        for i, arg in enumerate(argv):
            if i < len(argExplanations):
                print(argExplanations[i] + ':\t' + argv[i])
            else:
                print('arg {}: '.format(i) + argv[i])

        for j in range(i, 4):
            print(argExplanations[j])

        exit(-1)

    wpu.print_blue('MESSAGE: File name: ' + fname)
    wpu.set_at_ini_file(inifname, 'Files', 'Data', fname)

    if '/' in fname:
        fname4graphs = (fname.rsplit('/', 1)[0] + '/fit_' + fname.rsplit('/', 1)[1])
    else:
        fname4graphs = 'fit_' + fname

    fname4graphs = fname4graphs.replace('.csv', '')
    wpu.log_this('Input File : ' + fname, preffname=fname4graphs)

    wpu.set_at_ini_file(inifname, 'Parameters',
                        'photon energy [kev]', str(phenergy*1e-3))

    wpu.set_at_ini_file(inifname, 'Parameters',
                        'Ay for vibration [um]', str(a_y*1e6))

    results, _, _ = wpu.load_csv_file(fname)

    zvec = results[:, 0]

    contrast_V = results[:, 1]
    contrast_V -= np.min(contrast_V)
    contrast_H = results[:, 2]
    contrast_H -= np.min(contrast_H)

    pattern_period_Vert_z = results[:, 3]
    pattern_period_Horz_z = results[:, 4]

    if zvec[-1] - zvec[0] < 0:
        zvec = zvec[::-1]
        contrast_V = contrast_V[::-1]
        contrast_H = contrast_H[::-1]

    wpu.log_this(inifname=inifname)

    return (zvec,
            pattern_period_Vert_z, contrast_V,
            pattern_period_Horz_z, contrast_H,
            phenergy, a_y,
            fname4graphs,
            menu_options)


# %%
def fit_period_vs_z(zvec, pattern_period_z, contrast, direction,
                    threshold=0.005, fname4graphs='graph_'):

    args_for_NOfit = np.argwhere(contrast < threshold).flatten()
    args_for_fit = np.argwhere(contrast >= threshold).flatten()

    if 'Hor' in direction:
        ls1 = '-ro'
        lx = 'r'
        lc2 = 'm'
    else:
        ls1 = '-ko'
        lx = 'k'
        lc2 = 'c'

    plt.figure(figsize=(10, 7))
    plt.plot(zvec[args_for_NOfit]*1e3, pattern_period_z[args_for_NOfit]*1e6,
             'o', mec=lx, mfc='none', ms=8, label='not used for fit')
    plt.plot(zvec[args_for_fit]*1e3, pattern_period_z[args_for_fit]*1e6,
             ls1, label=direction)

    fit1d = np.polyfit(zvec[args_for_fit], pattern_period_z[args_for_fit], 1)
    sourceDistance = fit1d[1]/fit1d[0]
    patternPeriodFromData = fit1d[1]
    plt.plot(zvec[args_for_fit]*1e3,
             (fit1d[0]*zvec[args_for_fit] + fit1d[1])*1e6,
             '-', c=lc2, lw=2, label='Fit ' + direction)
    plt.text(np.min(zvec[args_for_fit])*1e3,
             np.min(pattern_period_z)*1e6,
             'source dist = {:.2f}m, '.format(fit1d[1]/fit1d[0]) +
             r'$p_o$ = {:.3f}um'.format(fit1d[1]*1e6),
             bbox=dict(facecolor=lc2, alpha=0.85))

    plt.xlabel(r'Distance $z$  [mm]', fontsize=14)
    plt.ylabel(r'Pattern Period [$\mu$m]', fontsize=14)
    plt.title('Pattern Period vs Detector distance, ' + direction,
              fontsize=14, weight='bold')

    plt.legend(fontsize=14, loc=1)

    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    return sourceDistance, patternPeriodFromData


# %%
def _chi2(data, model):
    eps = np.spacing(1)
    data_zerofree = np.where(np.abs(data) > eps, data, eps)
    return np.sum(np.abs(data_zerofree-model)/np.max(data_zerofree))/data.size


# %%
def fit_z_scan_talbot(zvec, contrast, wavelength,
                      patternPeriod, sourceDist,
                      fname4graphs):

    def _func_4_fit(z, Amp, z_period, sigma, phase, sourceDist):

        return Amp*np.abs(np.sin(z/z_period*np.pi/(1 + z/sourceDist) +
                                 phase*2*np.pi)) * \
                                 np.exp(-z**2/2/sigma**2/(1 + z/sourceDist)**2)

    p0 = [1.0, patternPeriod**2/wavelength, .96, 0.05, sourceDist]

    bounds_low = [1e-3, p0[1]*0.9999, .01,
                  -.1, np.min((0.5*sourceDist, 1.5*sourceDist))]

    bounds_up = [2.0,  p0[1]*1.0001, 10.,
                 .1, np.max((0.5*sourceDist, 1.5*sourceDist))]

    wpu.print_blue('Fit 1D')
    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0,
                           bounds=(bounds_low, bounds_up))

    ppatternFit = np.sqrt(wavelength*popt[1])
    results_Text = 'Fitting Results\n'
    results_Text += 'Grating Period: {:.2g} um\n'.format(ppatternFit*1e6)

    for i, parname in enumerate(['Amp', 'z_period [m]', 'sigma[m]',
                                 'phase [pi rad]', 'sourceDist [m]']):

        results_Text += parname + ' : ' + str('{:.6g}'.format(popt[i]) + '\n')

    cohLength = np.abs(popt[2])*wavelength/(ppatternFit)

    results_Text += 'Coherent length: {:.6g} um\n'.format(cohLength*1e6)
    _text_to_fig(results_Text, width=1.0)
    wpu.save_figs_with_idx(fname4graphs)

    fitted_curve = _func_4_fit(zvec, popt[0], popt[1],
                               popt[2], popt[3], popt[4])

    envelope = _func_4_fit(zvec, popt[0], 1e10,
                           popt[2], 1/4, popt[4])

    return fitted_curve, envelope, cohLength


# %%
def fit_z_scan_talbot_exp_dec(zvec, contrast, wavelength,
                      patternPeriod, sourceDist,
                      fname4graphs):

    def _func_4_fit(z, Amp, z_period, sigma, phase, sourceDist, alpha):


        return Amp*np.abs(np.sin(z/z_period*np.pi/(1 + z/sourceDist) +
                                 phase*2*np.pi)) * \
                                 np.exp(-z**2/2/sigma**2/(1 + z/sourceDist)**2) * \
                                 np.exp(-alpha/z)

    p0 = [1.0, patternPeriod**2/wavelength, .96, 0.05, sourceDist,.0054356345]

    bounds_low = [1e-3, p0[1]*0.9999, .01,
                  -.1, np.min((0.9*sourceDist, 1.1*sourceDist)), -10.0]

    bounds_up = [2.0,  p0[1]*1.0001, 10.,
                 .1, np.max((0.9*sourceDist, 1.1*sourceDist)), 10.0]

    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0,
                           bounds=(bounds_low, bounds_up))

    ppatternFit = np.sqrt(wavelength*popt[1])
    results_Text = 'Fitting Results\n'
    results_Text += 'Pattern Period: {:.2g} um\n'.format(ppatternFit*1e6)

    for i, parname in enumerate(['Amp', 'z_period [m]', 'sigma[m]',
                                 'phase [pi rad]', 'sourceDist [m]', 'alpha']):

        results_Text += parname + ' : ' + str('{:.6g}'.format(popt[i]) + '\n')

    cohLength = np.abs(popt[2])*wavelength/(ppatternFit)
    alpha = -popt[5]

    results_Text += 'Coherent length: {:.6g} um\n'.format(cohLength*1e6)
    _text_to_fig(results_Text, width=1.0)
    wpu.save_figs_with_idx(fname4graphs)

    fitted_curve = _func_4_fit(zvec, popt[0], popt[1],
                               popt[2], popt[3], popt[4], popt[5])

    envelope = _func_4_fit(zvec, popt[0], 1e10,
                           popt[2], 1/4, popt[4], popt[5])

    return fitted_curve, envelope, cohLength, alpha


# %%
def fit_z_scan_talbot2gauss(zvec, contrast, wavelength,
                            patternPeriod, sourceDist,
                            cohLength,
                            fname4graphs, title4graph='Title'):

    def _func_4_fit(z, Amp, Amp2, z_period, sigma1, sigma2, phase, sourceDist):
        return np.abs(np.sin(z/z_period*np.pi/(1 + z/sourceDist) +
                      phase*2*np.pi)) * (Amp*np.exp(-z**2/2/sigma1**2) +
                                         Amp2*np.exp(-z**2/2/sigma2**2))

    cohL_o = cohLength/wavelength*patternPeriod
    p0 = [1.0, .5, 2*patternPeriod**2/2/wavelength,
          cohL_o, cohL_o/3, .5, sourceDist]

    bounds = ([.0001, .0001, .9*2*patternPeriod**2/2/wavelength,
               cohL_o*0.9, cohL_o*0.01,
               -1., np.min((0.5*sourceDist, 1.5*sourceDist))],
              [2.0,  2.0, 1.1*2*patternPeriod**2/2/wavelength,
               cohL_o*20, cohL_o*0.9,
               1., np.max((0.5*sourceDist, 1.5*sourceDist))])

    wpu.print_blue('Fit 1D, Two Gaussians Envelopes')

    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0,
                           bounds=bounds)

    ppatternFit = np.sqrt(wavelength*popt[2])
    results_Text = 'Fitting Results\n'
    results_Text += 'Pattern Period: {:.2g} um\n'.format(ppatternFit*1e6)

    for i, parname in enumerate(['Amp1', 'Amp2', 'z_period [m]',
                                 'sigma1[m]', 'sigma2[m]',
                                 'phase [pi rad]', 'sourceDist [m]']):

        results_Text += parname + ' : ' + str('{:.6g}'.format(popt[i]) + '\n')

    cohLength1 = np.abs(popt[3])*wavelength/ppatternFit
    cohLength2 = np.abs(popt[4])*wavelength/ppatternFit

    results_Text += 'Coherent length1: {:.6g} um\n'.format(cohLength1*1e6)
    results_Text += 'Coherent length2: {:.6g} um\n'.format(cohLength2*1e6)
    _text_to_fig(results_Text, width=1.0)
    wpu.save_figs_with_idx(fname4graphs)

    fitted_curve = _func_4_fit(zvec, popt[0], popt[1],
                               popt[2], popt[3], popt[4],
                               popt[5], popt[6])

    envelope = _func_4_fit(zvec, popt[0], popt[1], 1e10,
                           popt[3], popt[4], 1/4, popt[6])

    return fitted_curve, envelope, cohLength1, cohLength2

# %%
#    plt.figure(figsize=(12, 9))
#    plt.plot(zvec*1e3, 100*contrast, '-ok', ms=7, label='data')
#    chi2 = _chi2(contrast, fitted_curve)
#    wpu.print_blue('chi2 Fit = {:.3f}'.format(chi2))
#
#    zvec_4_fit = np.linspace(zvec[0], zvec[-1], zvec.size*5)
#    plt.plot(zvec_4_fit*1e3,
#             100*_func_4_fit(zvec_4_fit, popt[0], popt[1], popt[2],
#                             popt[3], popt[4], popt[5]),
#             '-r', lw=3, label='Fit')
#
#    plt.plot(zvec*1e3,
#             100*(popt[0]*np.exp(-zvec**2/2/popt[3]**2)) +
#             100*(popt[1]*np.exp(-zvec**2/2/popt[4]**2)),
#             '-g', lw=3, label='2 Gaussians decay')
#
#    title4graph += r', $l_{coh}$ =' + ' {:.3f} um'.format(cohLength1*1e6)
#    title4graph += r', $l_{coh}$ =' + ' {:.3f} um'.format(cohLength2*1e6)
#    title4graph += r', $\chi^2$ = {:.3f}'.format(chi2)
#
#    plt.xlabel('Distance [mm]', fontsize=27)
#    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
#    plt.title(title4graph, fontsize=27, weight='bold')
#    plt.legend(fontsize=22)
#    # plt.grid()
#    wpu.save_figs_with_idx(fname4graphs)
#    plt.show(block=False)
#
#    return fitted_curve, envelope, cohLength1, cohLength2

# %%
def fit_z_scan_talbot_shaked(zvec, contrast, wavelength,
                             patternPeriod, sourceDist,
                             zero1,
                             fname4graphs):

    # np.exp(-1) = j0(1.32583)

    from scipy.special import j0

    def _func_4_fit(z, Amp, z_period, sigma, phase, sourceDist, zo):

        return Amp*np.abs(np.sin(z/z_period*np.pi/(1 + z/sourceDist) +
                                 phase*2*np.pi)) * \
                                 np.exp(-z**2/2/sigma**2/(1 + z/sourceDist)**2) * \
                                 np.abs(j0(z/(1 + z/sourceDist)*besselzero1/zo))

    p0 = [1.0, patternPeriod**2/wavelength, .96, 0.05, sourceDist, zero1]

    bounds_low = [1e-3, p0[1]*0.9999, .01,
                  -.1, np.min((0.9999*sourceDist, 1.0001*sourceDist)),
                  0.9*zero1]

    bounds_up = [2.0,  p0[1]*1.0001, 10.,
                 .1, np.max((0.9999*sourceDist, 1.0001*sourceDist)),
                 1.1*zero1]

    popt, pcov = curve_fit(_func_4_fit, zvec, contrast, p0=p0,
                           bounds=(bounds_low, bounds_up))

    results_Text = 'Fitting Results\n'

    for i, parname in enumerate(['Amp', 'z_period [m]', 'sigma[m]',
                                 'phase [pi rad]', 'sourceDist [m]',
                                 'Bessel 1st zero [m]']):

        results_Text += parname + ' : ' + str('{:.6g}'.format(popt[i]) + '\n')

    ppatternFit = np.sqrt(wavelength*popt[1])
    results_Text += 'Grating Period: {:.2g} um\n'.format(ppatternFit*1e6)
    sigma_bessel_V = popt[5]*1.32583/besselzero1
    cohLength_bessel = np.abs(sigma_bessel_V)*wavelength/(ppatternFit)

    cohLength = np.abs(popt[2])*wavelength/(ppatternFit)

    results_Text += 'Sigma Bessel: {:.6g} m\n'.format(sigma_bessel_V)
    results_Text += 'Coherent length Gauss env: {:.6g} um\n'.format(cohLength_bessel*1e6)
    results_Text += 'Coherent length Bessel env: {:.6g} um\n'.format(cohLength*1e6)
    _text_to_fig(results_Text, width=1.0)
    wpu.save_figs_with_idx(fname4graphs)

    fitted_curve = _func_4_fit(zvec, popt[0], popt[1],
                               popt[2], popt[3], popt[4], popt[5])

    envelope = 100 * (popt[0] *
                      np.exp(-zvec**2/2/popt[2]**2 / (1 + zvec/sourceDist)**2) *
                      np.abs(j0(zvec*besselzero1/popt[5])))

    return fitted_curve, envelope, cohLength_bessel, sigma_bessel_V


# %%
def plot_fit_z_scan(zvec, contrast, fitted_curve, envelope,
                    cohLength, fname4graphs, title4graph='Title'):

    if 'oriz' in title4graph:  # color line for horizontal
        linecolor = 'r'
    else:
        linecolor = 'k'  # color line for vertical and fallback condition

    plt.figure(figsize=(12, 9))
    plt.plot(zvec*1e3, 100*contrast, '-o', c=linecolor, ms=7, label='data')

    chi2 = _chi2(contrast, fitted_curve)
    wpu.print_blue('chi2 Fit = {:.3f}'.format(chi2))
    plt.plot(zvec*1e3,
             100*fitted_curve,
             '-c', lw=3, label='Fit')

    plt.plot(zvec*1e3,
             100*envelope,
             '-g', lw=3, label='Envelope to the fit')

    title4graph += r', $\chi^2$ = {:.3f}'.format(chi2)

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph, fontsize=27, weight='bold')
    plt.legend(fontsize=22)
    plt.ylim(ymax=1.1*np.max(100*contrast))
    wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)


# %%
def plot_several_envelopes(zvec, envelope_list, label_list, lf_list=['-'],
                           fname4graphs='graph.png', title4graph='Title'):

    plt.figure(figsize=(12, 9))

    if len(lf_list) == 1:
        lf_list = lf_list*len(envelope_list)

    for i, envelope in enumerate(envelope_list):

        plt.plot(zvec*1e3, envelope*100,
                 lf_list[i], lw=3, label=label_list[i])

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph, fontsize=27, weight='bold')
    plt.legend(fontsize=22)
    wpu.save_figs_with_idx(fname4graphs)

    plt.show(block=False)


# %%
def _text_to_fig(text, width=1.0):

    plt.figure()
    for i in range(text.count('\n')):
        plt.text(0, -i, text.split('\n')[i], fontsize=24)

    plt.ylim(-text.count('\n'), 0)
    plt.xlim(0, width)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())


# %%
def _load_data_from_pickle(fname, saveFlag=False):

    print('MESSAGE: Loading ' + fname)
    fig = pickle.load(open(fname, 'rb'))
    fig.set_size_inches((12, 9), forward=True)

    if saveFlag:
        wpu.save_figs_with_idx(fname4graphs)

    plt.show(block=False)  # this lines keep the script alive to see the plot

    curves = []

    for i in range(len(fig.axes[0].lines)):

        curves.append(np.asarray(fig.axes[0].lines[i].get_data()))

    return curves


# %%
def _extract_envelope(data, zvec, zperiod=1.0, fitInitialDistances=False,
                      saveGraphs=True, title4graph='Title'):
    '''
    This function extract the envelop of the z_scan by using some properties
    of the Fourier Transform

    Note that, because we cant meassure veru dshort distances down to zero, it
    is used a sine function to complete the data. This is the same as having a
    constant envelop.

    Also, because of the circular periodicity of the Discrete Fourier
    Transform, it is used a reflection of the
    '''

    new_max_z = (zvec[-1]//zperiod + 1)*zperiod
    zstep = np.average(np.diff(zvec))

    zvec = np.arange(zvec[0], new_max_z, zstep)

    data4fft = zvec*0.0

    data4fft[0:data.size] = data

    data4fft2 = data4fft*0.0
    min_args = argrelmin(data4fft, order=5)[0]
    mult_factor = 1

    for i in range(data4fft.size):

        if i in min_args:
            mult_factor = mult_factor*-1
            data4fft2[i] = 0

        else:
            data4fft2[i] = data4fft[i]*mult_factor

    if fitInitialDistances:
        # This avoids a sharp edge at the position zero. This is only
        # necessary because we cant meassure the values down to zero mm
        # WG: I recommend to use all the time

        dummy_Data = np.sin(np.arange(0, zvec[0], zstep)/zperiod*2*np.pi)
        dummy_Data *= data4fft[0]/dummy_Data[-1]
        dummy_Data = dummy_Data[:-1]  # remove last point

        #        print(dummy_Data)

        data4fft2 = np.concatenate((-1*data4fft2[::-1],
                                    -dummy_Data[::-1],
                                    dummy_Data[1:],
                                    data4fft2))

        z4plot = wpu.realcoordvec(data4fft2.size, zstep)

    else:

        data4fft2 = np.concatenate((-1*data4fft2[::-1], np.array([0.0]),
                                    data4fft2))

        z4plot = wpu.realcoordvec(data4fft2.size, zstep)

    # Plot 1
    plt.figure(figsize=(12, 9))
    plt.plot(zvec*1e3, data4fft,
             '-ok', lw=3, label='data')

    plt.plot(zvec[argrelmin(data4fft, order=5)]*1e3,
             data4fft[argrelmin(data4fft, order=5)],
             'om', lw=3)

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph + ', minimum ', fontsize=27, weight='bold')
    # plt.grid()
    plt.legend()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    # Plot 2
    plt.figure(figsize=(12, 9))
    plt.plot(z4plot*1e3, data4fft2,
             '-ok', lw=3)
    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph +
              r', data $\times$ square function, Function for FFT ',
              fontsize=27, weight='bold')
    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    # FFT
    fft_contrast = np.fft.fft(data4fft2)
    fvec = wpu.fouriercoordvec(data4fft2.shape[0], zstep)

    # zero padding

    #    fft_contrast = fft_contrast[fft_contrast.size // 2:-1]
    #    fvec = fvec[fvec.size // 2:-1]

    fft_contrast = np.pad(fft_contrast[0:fft_contrast.size // 2],
                          (0, fft_contrast.size - fft_contrast.size // 2),
                          'constant', constant_values=0.0)

    envelope = 2*np.abs(np.fft.ifft(fft_contrast))

    if fitInitialDistances:

        envelope = np.fft.ifftshift(envelope)[0:envelope.size//2 + dummy_Data.size]
        z4plot2 = np.linspace(0.0, zvec[-1], envelope.size)

    else:
        envelope = envelope[envelope.size//2:-1]
        z4plot2 = np.linspace(zvec[0], zvec[-1], envelope.size)

    # Plot 3

    plt.figure(figsize=(12, 9))
    plt.plot(fvec, np.abs(np.fft.fftshift(fft_contrast)), '-o')
    plt.xlabel('Spatial Frequency [1/m]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph + ', FFT', fontsize=27, weight='bold')
    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    # Plot 4

    plt.figure(figsize=(12, 9))

    plt.plot(zvec*1e3, data4fft,
             '-ok', lw=3, label='data')

    plt.plot(z4plot2*1e3, envelope, '-g', lw=3, label='Calculated Envelope')
    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph + ', Data and Calculated Envelope ',
              fontsize=27, weight='bold')
    plt.legend()
    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)

    plt.show(block=False)

    # Plot 5

    plt.figure(figsize=(12, 9))

    plt.plot(z4plot*1e3, data4fft2,
             '-ok', lw=3)

    plt.plot(z4plot2*1e3, envelope, '-g', lw=3)
    plt.plot(-z4plot2*1e3, envelope, '-g', lw=3)
    plt.plot(z4plot2*1e3, -envelope, '-g', lw=3)
    plt.plot(-z4plot2*1e3, -envelope, '-g', lw=3)

    plt.xlabel('Distance [mm]', fontsize=27)
    plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=27)
    plt.title(title4graph + ', Data and Calculated Envelope ',
              fontsize=27, weight='bold')
    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    return envelope, z4plot2


# %% Coh Function from Talbot envelope

def _coh_func_from_talbot_envelope(envelope, z_envelope, patternPeriod,
                                   title4graph='Title', saveGraphs=False):

    coh_function = np.concatenate((envelope[:0:-1], envelope))
    coh_func_coord = z_envelope*wavelength/patternPeriod
    coh_func_coord = np.concatenate((-coh_func_coord[:0:-1], coh_func_coord))

    plt.figure(figsize=(12, 9))
    plt.plot(coh_func_coord*1e6, coh_function/np.max(coh_function),
             '-og', lw=3)
    plt.xlabel(title4graph + r' Position [$\mu m$]', fontsize=27)
    plt.ylabel('Coh Function', fontsize=27)
    plt.title(title4graph + ' Coh Function', fontsize=27, weight='bold')

    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    return coh_function, coh_func_coord


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

    p0 = [10.000, besselzero1/zero1, 50e-6]

    arg4fit = np.where(np.abs(coh_func_coord) > 1e-6)

    popt, pcov = curve_fit(_func4fitCoh,
                           coh_func_coord[arg4fit],
                           coh_function[arg4fit], p0=p0)

    yamp = wavelength*sourceDistance*popt[1]/2/np.pi
    beam_size = wavelength*sourceDistance/popt[2]

    print('Fit bessel')
    print('Amp, y_o, FWHM beam')
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
    # plt.grid()
    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    return coh_func_coord, fitted_func


# %% Source calculation
def _source_from_coh_func(coh_function, coh_func_coord, sourceDist,
                          minOrd=5, graphLim=500,
                          title4graph='Title', saveGraphs=False):

    min_args = argrelmin(gaussian_filter(coh_function, 10), order=minOrd)[0]
    mult_factor = 1

    local_coh_function = coh_function*0.0

    for i in range(coh_function.size):

        if i in min_args:
            mult_factor = mult_factor*-1
            local_coh_function[i] = 0

        else:
            local_coh_function[i] = coh_function[i]*mult_factor

    local_coh_function *= np.sign(local_coh_function[np.argmax(np.abs(local_coh_function))])

    plt.figure(figsize=(12, 9))
    plt.plot(coh_func_coord*1e6, local_coh_function, '-ob', lw=3)
    plt.xlabel(title4graph + r' Source Position [$\mu m$]', fontsize=27)
    plt.ylabel('Source Profile [a.u.]', fontsize=27)

    plt.title(title4graph + ' New Coh Func',
              fontsize=27, weight='bold')

    # plt.grid()

    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    source_profile = np.abs(np.fft.ifftshift(np.fft.ifft(local_coh_function)))

    source_profile *= 1/np.max(source_profile)

    source_units = wpu.fouriercoordvec(source_profile.size,
                                       coh_func_coord[1] -
                                       coh_func_coord[0])*sourceDist*wavelength

    plt.figure(figsize=(12, 9))
    plt.plot(source_units*1e6, source_profile, '-ob', lw=3)
    plt.xlabel(title4graph + r' Source Position [$\mu m$]', fontsize=27)
    plt.ylabel('Source Profile [a.u.]', fontsize=27)

    FWHM_x = source_units[np.argmin(np.abs(source_profile - .50000))]
    plt.xlim([-graphLim, graphLim])

    plt.title(title4graph +
              ' Source Profile, FWHM={:.2f}um'.format(2*np.abs(FWHM_x)*1e6),
              fontsize=27, weight='bold')

    if saveGraphs:
        wpu.save_figs_with_idx(fname4graphs)
    plt.show(block=False)

    return source_profile, source_units


# %%

if __name__ == '__main__':
    '''
    To run in the command line use:
    fit_singleGratingCoherence_z_peaks.py arg1 arg2 arg3 arg4 arg5



    arg1:    photon Energy [KeV]
    arg2:    Grating Period [µm]
    arg3:    CB Grating Orientation: 'Edge' or 'Diagonal'
    arg4:    file name [.csv]

    '''

    (zvec_all,
     pattern_period_Vert_z, contrast_V_all,
     pattern_period_Horz_z, contrast_H_all,
     phenergy, a_y,
     fname4graphs,
     menu_options) = _load_experimental_pars(sys.argv)

    a_y *= 1/2*(1+66/27)*1e-6
    # above you need to convert the shaking
    # of the optical element to the shaking
    # of the source

    wavelength = hc/phenergy

    # %% Plot period to obtain divergence and source distance

    (sourceDistance_V,
     patternPeriodFromData_V) = fit_period_vs_z(zvec_all, pattern_period_Vert_z,
                                                contrast_V_all,
                                                direction='Vertical',
                                                threshold=.005,
                                                fname4graphs=fname4graphs)

    (sourceDistance_H,
     patternPeriodFromData_H) = fit_period_vs_z(zvec_all, pattern_period_Horz_z,
                                                contrast_H_all,
                                                direction='Horizontal',
                                                threshold=0.005,
                                                fname4graphs=fname4graphs)

    comments = 'Source dist in Vertical, from period change = '
    comments += '{:.4f}m\n'.format(sourceDistance_V)
    comments += 'Grating period Vertical = '
    comments += '{:.4f}um\n'.format(patternPeriodFromData_V*1e6)
    comments += 'Source dist in Horizont, from period change = '
    comments += '{:.4f}m\n'.format(sourceDistance_H)
    comments += 'Grating period Horizont = '
    comments += '{:.4f}um'.format(patternPeriodFromData_H*1e6)
    comments += '\nREMEMBER THAT THE ERROR IN THE MAGNIFICATION OF THE LENS '
    comments += 'IS AFFECTING THE GRATING PERIOD VALUES\n\n'

    wpu.log_this(comments)

    sourceDistance_V = np.min((np.abs(sourceDistance_V), 1e8))*np.sign(sourceDistance_V)
    sourceDistance_H = np.min((np.abs(sourceDistance_H), 1e8))*np.sign(sourceDistance_H)

# %% Select Maximum, minimum or mean value

    uniqueZ = np.unique(zvec_all)

    options2run = ['']

    if np.size(uniqueZ) == np.size(zvec_all):
        options2run = ['All Data']
    else:

        meanV = uniqueZ*0.0
        stdV = uniqueZ*0.0
        maxCv = uniqueZ*0.0
        minCv = uniqueZ*0.0

        meanH = uniqueZ*0.0
        stdH = uniqueZ*0.0
        maxCh = uniqueZ*0.0
        minCh = uniqueZ*0.0

        for i in range(uniqueZ.size):
            meanV[i] = np.mean(contrast_V_all[np.argwhere(zvec_all == uniqueZ[i])])
            maxCv[i] = np.max(contrast_V_all[np.argwhere(zvec_all == uniqueZ[i])])
            minCv[i] = np.min(contrast_V_all[np.argwhere(zvec_all == uniqueZ[i])])
            stdV[i] = np.std(contrast_V_all[np.argwhere(zvec_all == uniqueZ[i])])

            meanH[i] = np.mean(contrast_H_all[np.argwhere(zvec_all == uniqueZ[i])])
            maxCh[i] = np.max(contrast_H_all[np.argwhere(zvec_all == uniqueZ[i])])
            minCh[i] = np.min(contrast_H_all[np.argwhere(zvec_all == uniqueZ[i])])
            stdH[i] = np.std(contrast_H_all[np.argwhere(zvec_all == uniqueZ[i])])

        if menu_options & 0b1111 == 0:
            options2run = easyqt.get_list_of_choices('Select which values to fit',
                                                     ['All Data', 'Maximum',
                                                      'Minimum', 'Mean'])
        else:
            if menu_options & 0b1000 == 0b1000:
                options2run.append('All Data')
            if menu_options & 0b0100 == 0b0100:
                options2run.append('Maximum')
            if menu_options & 0b0010 == 0b0010:
                options2run.append('Minimum')
            if menu_options & 0b0001 == 0b0001:
                options2run.append('Mean')


# %%

    label_list_4plot = []
    envelope_V_list = []
    envelope_H_list = []

    envelope2gauss_V_list = []
    envelope2gauss_H_list = []

    for what2run in options2run:

        label_list_4plot.append(what2run)

        if 'Mean' in what2run:
            zvec = uniqueZ
            contrast_V = meanV
            contrast_H = meanH

        elif 'Maximum' in what2run:
            zvec = uniqueZ
            contrast_V = maxCv
            contrast_H = maxCh

        elif 'Minimum' in what2run:
            zvec = uniqueZ
            contrast_V = minCv
            contrast_H = minCh
        else:
            zvec = zvec_all
            contrast_V = contrast_V_all
            contrast_H = contrast_H_all
            pass

        # Main Fit Vertical
        if a_y < 0:
            (fitted_curve_V,
             envelope_V,
             cohLength_V) = fit_z_scan_talbot(zvec, contrast_V, wavelength,
                                              patternPeriodFromData_V,
                                              sourceDist=sourceDistance_V,
                                              fname4graphs=fname4graphs)

            title4graph = what2run + r', Vertical, '
            title4graph += r'$l_{coh}$ ='
            title4graph += ' {:.3f} um'.format(cohLength_V*1e6)
            plot_fit_z_scan(zvec, contrast_V, fitted_curve_V, envelope_V,
                            cohLength_V,
                            fname4graphs=fname4graphs,
                            title4graph=title4graph)

            beam_size_V = wavelength*sourceDistance_V/cohLength_V/2/np.pi

            wpu.log_this('Vertical Coh Length: {:.2f}um'.format(cohLength_V*1e6))
            wpu.log_this('Beam Size Vertical: {:.2f}um\n'.format(beam_size_V*1e6))

        else:
            # Fit Vertical with shake
            zero1 = patternPeriodFromData_V*np.abs(sourceDistance_V) * \
                    besselzero1/(2*np.pi*a_y)

            (fitted_curve_V,
             envelope_V,
             cohLength_V,
             sigma_bessel_V) = fit_z_scan_talbot_shaked(zvec, contrast_V,
                                                        wavelength,
                                                        patternPeriodFromData_V,
                                                        sourceDist=sourceDistance_V,
                                                        zero1=zero1,
                                                        fname4graphs=fname4graphs)

            title4graph = what2run + r', Vertical, '
            title4graph += r'$l_{coh}$ =' + ' {:.3f} um'.format(cohLength_V*1e6)
            plot_fit_z_scan(zvec, contrast_V, fitted_curve_V, envelope_V,
                            cohLength_V,
                            fname4graphs=fname4graphs,
                            title4graph=title4graph)

            beam_size_V = wavelength*sourceDistance_V/cohLength_V/2/np.pi

            zeroFromFit = sigma_bessel_V/1.32583*besselzero1

            a_y_fit = patternPeriodFromData_V*np.abs(sourceDistance_V) * \
                besselzero1/(2*np.pi*zeroFromFit)/(1/2*(1+66/27)*1e-6)

            wpu.log_this('Vertical Coh Length: {:.2f}um'.format(cohLength_V*1e6))
            wpu.log_this('Beam Size Vertical: {:.2f}um\n'.format(beam_size_V*1e6))

            wpu.log_this('Vibration From Fit: {:.2f}um\n'.format(a_y_fit))

        # Main Fit Horizontal
        if True:

            (fitted_curve_H,
             envelope_H,
             cohLength_H) = fit_z_scan_talbot(zvec, contrast_H, wavelength,
                                              patternPeriodFromData_H,
                                              sourceDist=sourceDistance_H,
                                              fname4graphs=fname4graphs)

            title4graph = what2run + r', Horizontal, '
            title4graph += r'$l_{coh}$ ='
            title4graph += ' {:.3f} um'.format(cohLength_H*1e6)
            plot_fit_z_scan(zvec, contrast_H, fitted_curve_H, envelope_H,
                            cohLength_H,
                            fname4graphs=fname4graphs,
                            title4graph=title4graph)

            beam_size_H = wavelength*sourceDistance_H/cohLength_H/2/np.pi

            wpu.log_this('Horizontal Coh Length: {:.2f}um'.format(cohLength_H*1e6))
            wpu.log_this('Beam Size Horizontal: {:.2f}um\n'.format(beam_size_H*1e6))

        envelope_V_list.append(envelope_V)
        envelope_H_list.append(envelope_H)

        plot_several_envelopes(zvec, [envelope_V, envelope_H],
                               lf_list=['-k', '-r'],
                               label_list=['Vertical', 'Horizontal'],
                               fname4graphs=fname4graphs,
                               title4graph=label_list_4plot[-1] + ', Vert and Horz envelopes')

        twoGaussianFit = (menu_options & 0b100000 == 0b100000)
        if twoGaussianFit is True:
            # twoGaussianFit Vertical
            (fitted_2gauss_V,
             envelope2gaus_V,
             cohLength_V1,
             cohLength_V2) = fit_z_scan_talbot2gauss(zvec, contrast_V,
                                                     wavelength,
                                                     patternPeriodFromData_V,
                                                     sourceDistance_V,
                                                     cohLength=cohLength_V,
                                                     fname4graphs=fname4graphs,
                                                     title4graph=what2run + ', Vertical')

            beam_sizeV1 = wavelength*sourceDistance_V/cohLength_V1/2/np.pi
            print('Beam Size Vertical1: {:.2f}um'.format(beam_sizeV1*1e6))

            beam_sizeV2 = wavelength*sourceDistance_V/cohLength_V2/2/np.pi
            print('Beam Size Vertical2: {:.2f}um'.format(beam_sizeV2*1e6))

            wpu.log_this('Vertical1 Coh Length: {:.2f}um'.format(cohLength_V1*1e6))
            wpu.log_this('Beam Size Vertical1: {:.2f}um'.format(beam_sizeV1*1e6))
            wpu.log_this('Vertical2 Coh Length: {:.2f}um'.format(cohLength_V2*1e6))
            wpu.log_this('Beam Size Vertical2: {:.2f}um\n'.format(beam_sizeV2*1e6))

            title4graph = what2run + r', Vertical, '
            title4graph += r'$l_{coh1}$ ='
            title4graph += ' {:.3f} um, '.format(cohLength_V1*1e6)
            title4graph += r'$l_{coh2}$ ='
            title4graph += ' {:.3f} um'.format(cohLength_V2*1e6)
            plot_fit_z_scan(zvec, contrast_V, fitted_2gauss_V, envelope2gaus_V,
                            cohLength_V,
                            fname4graphs=fname4graphs,
                            title4graph=title4graph)

            # twoGaussianFit Horizontal
            (fitted_2gauss_H,
             envelope2gaus_H,
             cohLength_H1,
             cohLength_H2) = fit_z_scan_talbot2gauss(zvec, contrast_H,
                                                     wavelength,
                                                     patternPeriodFromData_H,
                                                     sourceDistance_H,
                                                     cohLength=cohLength_H,
                                                     fname4graphs=fname4graphs,
                                                     title4graph=what2run + ', Horizontal')

            beam_sizeH1 = wavelength*sourceDistance_H/cohLength_H1/2/np.pi
            print('Beam Size Horizontal1: {:.2f}um'.format(beam_sizeH1*1e6))

            beam_sizeH2 = wavelength*sourceDistance_H/cohLength_H2/2/np.pi
            print('Beam Size Horizontal2: {:.2f}um'.format(beam_sizeH2*1e6))

            wpu.log_this('Horizontal1 Coh Length: {:.2f}um'.format(cohLength_H1*1e6))
            wpu.log_this('Beam Size Horizontal1: {:.2f}um\n'.format(beam_sizeH1*1e6))
            wpu.log_this('Horizontal2 Coh Length: {:.2f}um'.format(cohLength_H2*1e6))
            wpu.log_this('Beam Size Horizontal2: {:.2f}um\n'.format(beam_sizeH2*1e6))

            title4graph = what2run + r', Horizontal, '
            title4graph += r'$l_{coh1}$ ='
            title4graph += ' {:.3f} um, '.format(cohLength_H1*1e6)
            title4graph += r'$l_{coh2}$ ='
            title4graph += ' {:.3f} um'.format(cohLength_H2*1e6)
            plot_fit_z_scan(zvec, contrast_H, fitted_2gauss_H, envelope2gaus_H,
                            cohLength_H,
                            fname4graphs=fname4graphs,
                            title4graph=title4graph)

            envelope2gauss_V_list.append(envelope2gaus_V)
            envelope2gauss_H_list.append(envelope2gaus_H)

            plot_several_envelopes(zvec, [envelope_V, envelope_H],
                                   lf_list=['-k', '-r'],
                                   label_list=['Vertical', 'Horizontal'],
                                   fname4graphs=fname4graphs,
                                   title4graph=label_list_4plot[-1] +
                                   ', Vert and Horz envelopes, 2-gauss')

# %% fit talbot times exponential decay for detector gain

    if False:

        (fitted_curve_V,
         envelope_V,
         cohLength_V,
         alpha_V) = fit_z_scan_talbot_exp_dec(zvec, contrast_V, wavelength,
                                              patternPeriodFromData_V,
                                              sourceDist=sourceDistance_V,
                                              fname4graphs=fname4graphs)

        title4graph = what2run + ', Vertical, Exponential Decay\n'
        title4graph += r'$l_{coh}$ ='
        title4graph += ' {:.3f} um'.format(cohLength_V*1e6)
        title4graph += r', $\alpha$ = {:.3f} um'.format(alpha_V)
        plot_fit_z_scan(zvec, contrast_V, fitted_curve_V, envelope_V,
                        cohLength_V,
                        fname4graphs=fname4graphs,
                        title4graph=title4graph)

        beam_size_V = wavelength*sourceDistance_V/cohLength_V/2/np.pi

        wpu.log_this('Vertical Coh Length: {:.2f}um'.format(cohLength_V*1e6))
        wpu.log_this('Beam Size Vertical: {:.2f}um'.format(beam_size_V*1e6))
        wpu.log_this('alpha, exp decay: {:.2f}um\n'.format(alpha_V))

# %% HA

    if (menu_options & 0b100000 == 0b100000):

        plot_several_envelopes(zvec, envelope2gauss_V_list,
                               label_list=label_list_4plot,
                               fname4graphs=fname4graphs,
                               title4graph='Vert envelopes, 2-gaussians fit')

        plot_several_envelopes(zvec, envelope2gauss_H_list,
                               label_list=label_list_4plot,
                               fname4graphs=fname4graphs,
                               title4graph='Hor envelopes, 2-gaussians fit')

    elif (menu_options & 0b000111 == 0b000111):

        plot_several_envelopes(zvec, envelope_V_list,
                               label_list=label_list_4plot,
                               fname4graphs=fname4graphs,
                               title4graph='Vert envelopes')

        plot_several_envelopes(zvec, envelope_H_list,
                               label_list=label_list_4plot,
                               fname4graphs=fname4graphs,
                               title4graph='Hor envelopes')


###############################################################################
# %% Sandbox
###############################################################################

# %% From here we have the part for extrating the source

    retrieveDOCfunc = False
    retrieveSource = False

    if retrieveDOCfunc is True:

        # %% extract DOC envelop Vertical
        envelopeV, z_envelopeV = _extract_envelope(contrast_V, zvec,
                                                   zperiod=zperiodFit*2,
                                                   fitInitialDistances=True,
                                                   saveGraphs=True,
                                                   title4graph='Vertical')

        [coh_functionV,
         coh_func_coordV] = _coh_func_from_talbot_envelope(envelopeV, z_envelopeV,
                                                           title4graph=what2run + ', Vertical',
                                                           saveGraphs=True)

        _ = _coh_func_fit_bessel(coh_functionV, coh_func_coordV,
                                 wavelength, sourceDistance_V,
                                 title4graph=what2run + ', Vertical', saveGraphs=True)

        # %% extract DOC envelop Vertical

        envelopeH, z_envelopeH = _extract_envelope(contrast_H, zvec,
                                                   zperiod=zperiodFit*2,
                                                   fitInitialDistances=True,
                                                   saveGraphs=True,
                                                   title4graph=what2run + ', Horizontal')

        [coh_functionH,
         coh_func_coordH] = _coh_func_from_talbot_envelope(envelopeH, z_envelopeH,
                                                           title4graph=what2run + ', Horizontal',
                                                           saveGraphs=True)

        _coh_func_fit_bessel(coh_functionH, coh_func_coordH,
                             wavelength, sourceDistance_H,
                             title4graph=what2run + ', Horizontal',
                             saveGraphs=True)

        # %% Calculate Source from DOC values

    if retrieveSource is True:

        # pad
        coh_functionV2 = np.pad(coh_functionV, mode='edge',
                                pad_width=(coh_functionV.size*2,
                                           coh_functionV.size*2))

        _source_from_coh_func(coh_functionV2,
                              np.linspace(coh_func_coordV[0]*5,
                                          coh_func_coordV[-1]*5,
                                          coh_functionV2.size),
                              sourceDistance_V,
                              minOrd=100, graphLim=10000,
                              title4graph=what2run + ', Vertical', saveGraphs=True)

        # % pad

        coh_functionH2 = np.pad(coh_functionH, mode='edge',
                                pad_width=(coh_functionH.size*2,
                                           coh_functionH.size*2))

        _source_from_coh_func(coh_functionH2,
                              np.linspace(coh_func_coordH[0]*5,
                                          coh_func_coordH[-1]*5,
                                          coh_functionH2.size),
                              sourceDistance_H,
                              minOrd=100,
                              title4graph=what2run + ', Horizontal',
                              saveGraphs=True)

    # %%

    delta = wpu.get_delta(8000, material='Be')[0]
