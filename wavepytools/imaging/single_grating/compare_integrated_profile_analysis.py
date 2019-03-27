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


'''

# %%% imports cell

import sys

if len(sys.argv) != 1:
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects

from wavepy.utils import easyqt
import wavepy.utils as wpu

import sys
import glob

wpu._mpl_settings_4_nice_graphs()


# %%
import os
def _intial_gui_setup(sys_argv):

    global inifname  # name of .ini file
    inifname = os.curdir + '/.' + os.path.basename(__file__).replace('.py', '.ini')

    for i, argv in enumerate(sys_argv):
        print('arg {}: '.format(i) + argv)

    if len(sys_argv) == 1:

        default_ini = wpu.load_ini_file(inifname)
        foldername = default_ini['Files']['Folder Name']

        if easyqt.get_yes_or_no('Select new folder?\n' +
                                '[ESC load folder of previous run]'):

            foldername = easyqt.get_directory_name(title='Select Folder that' +
                                                   '\ncontains *csv files')

    elif len(sys_argv) == 2:

        foldername = sys_argv[1]

    else:

        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n' +
              os.path.basename(__file__) + ' : (no inputs) load dialogs \n'
              '\n' +
              os.path.basename(__file__) + ' [args] \n'
              '\n'
              'arg1: folder that contains the *csv files\n'
              '\n')

        exit(-1)

    wpu.set_at_ini_file(inifname, 'Files', 'Folder Name', foldername)

    list_of_files = sorted(glob.glob(foldername + '/*csv'))

    nfiles = len(list_of_files)

    data, header_list = wpu.load_csv_file(list_of_files[0])

    wpu.print_blue('MESSAGE: Header of data files:')
    wpu.print_blue(header_list)

    ncurves = data.shape[1] - 1

    label_list = [fname.rsplit('_', 1)[1].split('.')[0]
                  for fname in list_of_files]

    #[fname.rsplit('/', 1)[1].split('_Y_integrated')[0]
    #                  for fname in list_of_files]



    return list_of_files, nfiles, ncurves, label_list, header_list


# %% Main

if __name__ == '__main__':

    # %%
    [list_of_files, nfiles, ncurves,
     label_list, header_list] = _intial_gui_setup(sys.argv)

    saveFlag = True
    step_files = 1
    addThisToTitle = '\nVertical Profile, Projected WF, grazing_angle = 2.6 mrad'

    # %%

    xmax = 0
    xmin = 1e6

    list_all_data = []

    for fname in list_of_files:

        data, _ = wpu.load_csv_file(fname)

        #        data[:,1:] *= np.pi

        list_all_data.append(data)

        xmin = np.min([xmin, np.min(data[:, 0])])

        xmax = np.max([xmax, np.max(data[:, 0])])

    # cant convert list_all_data to array because each array in the list can
    # have different sizes. It is necessary to do the spline to have same
    # vector sizes

    # %%
    lstyle_cycle, lc_cycle = wpu.line_style_cycle(ls=['-', '--'],
                                                  ms=['s', 'o', 'd', '^', 'v'],
                                                  cmap_str='default',
                                                  ncurves=nfiles)

    # %% Plot Raw data

    curve_index = 3

    plt.figure(figsize=(12, 12*9/16))

    for i in range(0, nfiles, step_files):

        data = list_all_data[i]

        plt.plot(data[:, 0]*1e3, data[:, curve_index]*1e9,
                 next(lstyle_cycle), color=next(lc_cycle), label=label_list[i])

    plt.xlabel('y [$mm$]')
    plt.ylabel('Height [$nm$]')
    plt.title(addThisToTitle)
    plt.legend(loc=0, fontsize=10)
    if saveFlag:
        wpu.save_figs_with_idx()
    plt.show()

    # %% splines

    from scipy.interpolate import interp1d

    fspline = []
    for _ in list_of_files:

        fspline.append([])

    for k in range(nfiles):

        data = list_all_data[k]

        for j in range(1, ncurves):

            data_xvec = data[:, 0] - data[0, 0]

            fspline[k].append(interp1d(data_xvec,
                                       data[:, curve_index],
                                       kind='cubic'))

    # %% use spline to create equally spaced matrix

    xstep = (xmax-xmin)/400
    expoent_step = (np.log10(xstep)//1)  # round step
    xstep = np.round(xstep/10**expoent_step)*10**expoent_step  # round step

    np.round(1.4325325, 1)
    xmargin = xstep*3
    xvec = np.arange(np.round(xmin, 6) + xmargin,
                     np.round(xmax, 6) - xmargin, xstep)

    xvec -= np.min(xvec) - xmargin

    allData = np.zeros((nfiles, xvec.size, ncurves - 1))

    for k in range(nfiles):
        for j in range(0, ncurves-1):
            allData[k, :, j] = fspline[k][j](xvec)

    # %% Plot Data

    ref_idx = 0 #nfiles -1  # index of reference in list_of_files

    # %%


    lstyle_cycle, lc_cycle = wpu.line_style_cycle(ls=['-', '--'],
                                                  ms=['s', 'o', 'd', '^', 'v'],
                                                  cmap_str='default',
                                                  ncurves=len(list_of_files))

    plt.figure(figsize=(12, 12*9/16))

    for k in range(0, nfiles, step_files):

        plt.plot(xvec*1e3, allData[k, :, curve_index]*1e9,
                 next(lstyle_cycle), color=next(lc_cycle),
                 label=label_list[k])

    # fit and plot in the same graph

    coefFit = np.polyfit(xvec, allData[ref_idx, :, 0], 2)

    fitData = xvec**2*coefFit[0] + xvec*coefFit[1] + coefFit[2]

    plt.plot(xvec*1e3, fitData*1e9, '--r',
             label='Fit ' + label_list[ref_idx])

    plt.title('Spline')
    plt.xlabel(r'y [$m m$]')
    plt.ylabel('Height [$nm$]')
    plt.legend(loc=0, fontsize=12)

    if saveFlag:
        wpu.save_figs_with_idx()
    plt.show()

    # %% Residual

    lstyle_cycle, lc_cycle = wpu.line_style_cycle(ls=['-', '--'],
                                                  ms=['s', 'o', 'd', '^', 'v'],
                                                  cmap_str='default',
                                                  ncurves=len(list_of_files))

    plt.figure(figsize=(12, 12*9/16))

    offset = 0

    for k in range(0, nfiles, step_files):

        plt.plot(xvec*1e3, (allData[k, :, curve_index] - fitData)*1e9 + offset*k,
                 next(lstyle_cycle), color=next(lc_cycle),
                 label=label_list[k])

    plt.title('Residual, Spline, Reference Fit from ' +
              label_list[ref_idx] + addThisToTitle)
    plt.xlabel(r'y [$m m$]')
    plt.ylabel('Height [$nm$]')
    plt.legend(loc=0, fontsize=12)

    if saveFlag:
        wpu.save_figs_with_idx()
    plt.show()

    # %% plot differences

    plt.figure(figsize=(12, 12*9/16))

    lstyle_cycle, lc_cycle = wpu.line_style_cycle(ls=['-', '--'],
                                                  ms=['s', 'o', 'd', '^', 'v'],
                                                  cmap_str='default',
                                                  ncurves=len(list_of_files))
    refData = allData[ref_idx, :, curve_index]

    for k in range(0, nfiles, step_files):

        plt.plot(xvec*1e3,
                 (allData[k, :, curve_index] - refData)*1e9,
                 next(lstyle_cycle), color=next(lc_cycle),
                 label=label_list[k])
                 # next(lstyle_cycle), label='diff ' + list_of_files[i][3:5] +'-2')

    plt.title("Height Difference, reference " +
              label_list[ref_idx] + addThisToTitle)
    plt.xlabel(r'y [$m m$]')
    plt.ylabel('Height [$nm$]')
    plt.legend(loc=0, fontsize=12)

    if saveFlag:
        wpu.save_figs_with_idx()
    plt.show()

    # %% plot differences


    lstyle_cycle, lc_cycle = wpu.line_style_cycle(ls=['-', '--'],
                                              ms=['s', 'o', 'd', '^', 'v'],
                                              cmap_str='default',
                                              ncurves=len(list_of_files))

    plt.figure(figsize=(12, 12*9/16))

    for k in range(0, nfiles, step_files):


        if k == 0:
            ydata = allData[k, :, curve_index] - allData[0, :, curve_index]
        else:
            ydata = allData[k, :, curve_index] - allData[k-1, :, curve_index]


        color=next(lc_cycle)
        plt.plot(xvec*1e3,
                 ydata*1e9,
                 next(lstyle_cycle), color=color,
                 label=label_list[k])
                 # next(lstyle_cycle), label='diff ' + list_of_files[i][3:5] +'-2')

        if nfiles/step_files < 15:
            pos = xvec.shape[0] // (nfiles+1) * k
            plt.text(xvec[pos]*1e3,
                     ydata[pos]*1e9, label_list[k],
                     fontsize=10, color='w',
                     bbox=dict(facecolor=color),
                     path_effects=[PathEffects.withStroke(linewidth=2,foreground="k")])

    #ylim = 40.0e-9
    #plt.xlim((xvec[0]*1e3 - 5, xvec[-1]*1e3 + 5))
    #plt.ylim((-ylim*1e9 - 5, ylim*1e9 + 5))
    plt.title("Height Difference, compared previous data" + addThisToTitle)
    plt.xlabel(r'y [$m m$]')
    plt.ylabel('Height [$nm$]')
    plt.legend(loc=0, fontsize=12)

    if saveFlag:
        wpu.save_figs_with_idx()
    #plt.close()
    plt.show(block=False)


