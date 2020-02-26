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
import os

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from wavepy.utils import easyqt
import wavepy.utils as wpu

from scipy.ndimage.filters import uniform_filter1d
from matplotlib.patches import Rectangle

wpu._mpl_settings_4_nice_graphs(otheroptions={'axes.formatter.use_mathtext':True,
                                              'axes.formatter.limits': '-3, 4'})


# %%
def _n_profiles_H_V(arrayH, arrayV, virtual_pixelsize,
                    zlabel=r'z',
                    titleH='Horiz', titleV='Vert',
                    nprofiles=5, filter_width=0,
                    remove1stOrderDPC=False,
                    saveFileSuf='',
                    saveFigFlag=True):

    xxGrid, yyGrid = wpu.grid_coord(arrayH, virtual_pixelsize)

    fit_coefs = [[], []]
    data2saveH = None
    data2saveV = None
    labels_H = None
    labels_V = None

    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['lines.linewidth'] = 2

    # Horizontal
    if np.all(np.isfinite(arrayH)):

        plt.figure(figsize=(12, 12*9/16))

        xvec = xxGrid[0, :]
        data2saveH = np.c_[xvec]
        header = ['x [m]']

        if filter_width != 0:
            arrayH_filtered = uniform_filter1d(arrayH, filter_width, 0)
        else:
            arrayH_filtered = arrayH

        ls_cycle, lc_jet = wpu.line_style_cycle(['-'], ['o', 's', 'd', '^'],
                                                ncurves=nprofiles,
                                                cmap_str='gist_rainbow_r')

        lc = []
        labels_H = []
        for i, row in enumerate(np.linspace(filter_width//2,
                                            np.shape(arrayV)[0]-filter_width//2-1,
                                            nprofiles + 2, dtype=int)):

            if i == 0 or i == nprofiles + 1:
                continue

            yvec = arrayH_filtered[row, :]

            lc.append(next(lc_jet))
            p01 = np.polyfit(xvec, yvec, 1)
            fit_coefs[0].append(p01)

            if remove1stOrderDPC:
                yvec -= p01[0]*xvec + p01[1]

            plt.plot(xvec*1e6, yvec, next(ls_cycle), color=lc[i-1],
                     label=str(row))

            if not remove1stOrderDPC:
                plt.plot(xvec*1e6, p01[0]*xvec + p01[1], '--',
                         color=lc[i-1], lw=3)

            data2saveH = np.c_[data2saveH, yvec]
            header.append(str(row))
            labels_H.append(str(row))

        if remove1stOrderDPC:
            titleH = titleH + ', 2nd order removed'
        plt.legend(title='Pixel Y', loc=0, fontsize=12)

        plt.xlabel(r'x [$\mu m$]', fontsize=18)
        plt.ylabel(zlabel, fontsize=18)
        plt.title(titleH + ', Filter Width = {:d} pixels'.format(filter_width),
                  fontsize=20)

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf + '_H')

        plt.show(block=False)

        header.append(zlabel + ', Filter Width = {:d} pixels'.format(filter_width))

        wpu.save_csv_file(data2saveH,
                          wpu.get_unique_filename(saveFileSuf +
                                                  '_WF_profiles_H', 'csv'),
                          headerList=header)

        plt.figure(figsize=(12, 12*9/16))
        plt.imshow(arrayH, cmap='RdGy',
                   vmin=wpu.mean_plus_n_sigma(arrayH, -3),
                   vmax=wpu.mean_plus_n_sigma(arrayH, 3))
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title(titleH + ', Profiles Position')

        currentAxis = plt.gca()

        _, lc_jet = wpu.line_style_cycle(['-'], ['o', 's', 'd', '^'],
                                         ncurves=nprofiles,
                                         cmap_str='gist_rainbow_r')

        for i, row in enumerate(np.linspace(filter_width//2,
                                            np.shape(arrayV)[0]-filter_width//2-1,
                                            nprofiles + 2, dtype=int)):

            if i == 0 or i == nprofiles + 1:
                continue

            currentAxis.add_patch(Rectangle((-.5, row - filter_width//2 - .5),
                                            np.shape(arrayH)[1], filter_width,
                                            facecolor=lc[i-1], alpha=.5))
            plt.axhline(row, color=lc[i-1])

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf + '_H')

        plt.show(block=True)

    # Vertical
    if np.all(np.isfinite(arrayV)):

        plt.figure(figsize=(12, 12*9/16))

        xvec = yyGrid[:, 0]
        data2saveV = np.c_[xvec]
        header = ['y [m]']

        if filter_width != 0:
            arrayV_filtered = uniform_filter1d(arrayV, filter_width, 1)
        else:
            arrayV_filtered = arrayV

        ls_cycle, lc_jet = wpu.line_style_cycle(['-'], ['o', 's', 'd', '^'],
                                                ncurves=nprofiles,
                                                cmap_str='gist_rainbow_r')

        lc = []
        labels_V = []
        for i, col in enumerate(np.linspace(filter_width//2,
                                            np.shape(arrayH)[1]-filter_width//2-1,
                                            nprofiles + 2, dtype=int)):

            if i == 0 or i == nprofiles + 1:
                continue

            yvec = arrayV_filtered[:, col]

            lc.append(next(lc_jet))
            p10 = np.polyfit(xvec, yvec, 1)
            fit_coefs[1].append(p10)

            if remove1stOrderDPC:
                yvec -= p10[0]*xvec + p10[1]

            plt.plot(xvec*1e6, yvec, next(ls_cycle), color=lc[i-1],
                     label=str(col))

            if not remove1stOrderDPC:
                plt.plot(xvec*1e6, p10[0]*xvec + p10[1], '--',
                         color=lc[i-1], lw=3)

            data2saveV = np.c_[data2saveV, yvec]
            header.append(str(col))
            labels_V.append(str(col))

        if remove1stOrderDPC:
            titleV = titleV + ', 2nd order removed'

        plt.legend(title='Pixel X', loc=0, fontsize=12)

        plt.xlabel(r'y [$\mu m$]', fontsize=18)
        plt.ylabel(zlabel, fontsize=18)

        plt.title(titleV + ', Filter Width = {:d} pixels'.format(filter_width),
                  fontsize=20)
        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf + '_Y')
        plt.show(block=False)

        header.append(zlabel + ', Filter Width = {:d} pixels'.format(filter_width))

        wpu.save_csv_file(data2saveV,
                          wpu.get_unique_filename(saveFileSuf +
                                                  '_WF_profiles_V', 'csv'),
                          headerList=header)

        plt.figure(figsize=(12, 12*9/16))
        plt.imshow(arrayV, cmap='RdGy',
                   vmin=wpu.mean_plus_n_sigma(arrayV, -3),
                   vmax=wpu.mean_plus_n_sigma(arrayV, 3))
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title(titleV + ', Profiles Position')

        currentAxis = plt.gca()

        for i, col in enumerate(np.linspace(filter_width//2,
                                            np.shape(arrayH)[1]-filter_width//2-1,
                                            nprofiles + 2, dtype=int)):

            if i == 0 or i == nprofiles + 1:
                continue


            currentAxis.add_patch(Rectangle((col - filter_width//2 - .5, -.5),
                                            filter_width, np.shape(arrayV)[0],
                                            facecolor=lc[i-1], alpha=.5))
            plt.axvline(col, color=lc[i-1])

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf + '_Y')

        plt.show(block=True)

    return data2saveH, data2saveV, labels_H, labels_V, fit_coefs


# %%

def integrate_DPC_cumsum(data_DPC, wavelength,
                         grazing_angle=0.0, projectionFromDiv=1.0,
                         remove2ndOrder=False,
                         labels=[],
                         xlabel='x', ylabel='Height',
                         titleStr='', saveFileSuf=''):

    ls_cycle, lc_cycle = wpu.line_style_cycle(['-'], ['o', 's', 'd', '^'],
                                              ncurves=data_DPC.shape[1] - 1,
                                              cmap_str='gist_rainbow_r')

    if grazing_angle//.00001 > 0:
        projection = 1/np.sin(grazing_angle)*projectionFromDiv
    else:
        projection = projectionFromDiv

    xvec = data_DPC[:, 0]*projection

    plt.figure(figsize=(12, 12*9/16))
    list_integrated = [xvec]

    header = [xlabel + ' [m]']

    for j_line in range(1, data_DPC.shape[1]):

        integrated = (np.cumsum(data_DPC[:, j_line] - np.mean(data_DPC[:, j_line]))
                      * (xvec[1]-xvec[0])) # TODO: removed mean 20181020

        #        integrated = (np.cumsum(data_DPC[:, j_line])) * (xvec[1]-xvec[0])

        integrated *= -1/2/np.pi*wavelength*np.abs(projection)

        p02 = np.polyfit(xvec, integrated, 2)
        fitted_pol2 = p02[0]*xvec**2 + p02[1]*xvec + p02[2]

        if remove2ndOrder:

            integrated -= fitted_pol2
            titleStr += 'Removed 2nd order, '

        # TODO: check here!!

        if j_line == 1:
            factor_x, unit_x = wpu.choose_unit(xvec)
            factor_y, unit_y = wpu.choose_unit(integrated)

        list_integrated.append(integrated)
        header.append(labels[j_line - 1])

        lc = next(lc_cycle)
        plt.plot(xvec*factor_x,
                 integrated*factor_y,
                 next(ls_cycle), c=lc,
                 label=labels[j_line - 1])

        if not remove2ndOrder:
            plt.plot(xvec*1e6, (fitted_pol2)*factor_y,
                     '--', color=lc, lw=3)

    marginx = 0.1*np.ptp(xvec*factor_x)
    plt.xlim([np.min(xvec*factor_x)-marginx,
              np.max(xvec*factor_x)+marginx])
    plt.xlabel(xlabel + r' [$' + unit_x + ' m$]')
    plt.ylabel(ylabel + r' [$' + unit_y + ' m$]')
    plt.legend(loc=0, fontsize=12)

    if grazing_angle//.00001 > 0:

        plt.title(titleStr + 'Mirror Height,\n' +
                  'grazing angle {:.2f} mrad,\n'.format(grazing_angle*1e3) +
                  'projection due divergence = ' +
                  r'$ \times $ {:.2f}'.format(projectionFromDiv))
    else:
        plt.title(titleStr + 'Integration Cumulative Sum')

    plt.tight_layout()
    wpu.save_figs_with_idx(saveFileSuf)
    plt.show()

    data2saveV = np.asarray(list_integrated).T

    header.append(ylabel + ' [m]')

    if grazing_angle//.00001 > 0:
        header.append('grazing_angle = {:.4g}'.format(grazing_angle))

    if projectionFromDiv//1 != 1:
        header.append('projection due divergence = ' +
                      '{:.2f}x'.format(projectionFromDiv))

    wpu.save_csv_file(data2saveV,
                      wpu.get_unique_filename(saveFileSuf +
                                              '_integrated_' + xlabel, 'csv'),
                      headerList=header)

    return np.asarray(list_integrated).T


# %%

def curv_from_height(height, virtual_pixelsize,
                     grazing_angle=0.0, projectionFromDiv=1.0,
                     labels=[],
                     xlabel='x', ylabel='Curvature',
                     titleStr='', saveFileSuf=''):

    ls_cycle, lc_cycle = wpu.line_style_cycle(['-'], ['o', 's', 'd', '^'],
                                              ncurves=height.shape[1] - 1,
                                              cmap_str='gist_rainbow_r')

    if grazing_angle//.00001 > 0:
        projection = 1/np.sin(grazing_angle)*projectionFromDiv
    else:
        projection = projectionFromDiv

    projected_pixel = virtual_pixelsize*projection
    xvec = wpu.realcoordvec(height.shape[0]-2, projected_pixel)

    print('projected_pixel')
    print(projected_pixel)

    plt.figure(figsize=(12, 12*9/16))
    list_curv = [xvec]

    header = [xlabel + ' [m]']

    for j_line in range(1, height.shape[1]):

        curv = np.diff(np.diff(height[:, j_line]))/projected_pixel**2

        if j_line == 1:
            factor_x, unit_x = wpu.choose_unit(xvec)

            #factor_y, unit_y = wpu.choose_unit(curv)

        list_curv.append(curv)
        header.append(labels[j_line - 1])

        plt.plot(xvec*factor_x, curv,
                 next(ls_cycle), c=next(lc_cycle),
                 label=labels[j_line - 1])

    marginx = 0.1*np.ptp(xvec*factor_x)
    plt.xlim([np.min(xvec*factor_x)-marginx,
              np.max(xvec*factor_x)+marginx])
    plt.xlabel(xlabel + r' [$' + unit_x + ' m$]')
    plt.ylabel(ylabel + r'[$m^{-1}$]')
    plt.legend(loc=0, fontsize=12)

    if grazing_angle//.00001 > 0:

        plt.title(titleStr + 'Mirror Curvature,\n' +
                  'grazing angle {:.2f} mrad,\n'.format(grazing_angle*1e3) +
                  'projection due divergence = ' +
                  r'$ \times $ {:.2f}'.format(projectionFromDiv))
    else:
        plt.title(titleStr + 'Curvature')

    plt.tight_layout()
    wpu.save_figs_with_idx(saveFileSuf)
    plt.show()

    data2saveV = np.asarray(list_curv).T

    header.append(ylabel + ' [1/m]')

    if grazing_angle//.00001 > 0:
        header.append(', grazing_angle = {:.4g}'.format(grazing_angle))

    if projectionFromDiv//1 != 1:
        header.append('projection due divergence = ' +
                      '{:.2f}x'.format(projectionFromDiv))

    wpu.save_csv_file(data2saveV,
                      wpu.get_unique_filename(saveFileSuf +
                                              '_curv_' + xlabel, 'csv'),
                      headerList=header)

    return np.asarray(list_curv).T

# %%


def _intial_gui_setup(sys_argv):

    global inifname  # name of .ini file
    inifname = os.curdir + '/.' + os.path.basename(__file__).replace('.py', '.ini')

    for i, argv in enumerate(sys_argv):
        print('arg {}: '.format(i) + argv)

    if len(sys_argv) == 1:

        default_ini = wpu.load_ini_file(inifname)
        p0 = float(default_ini['Parameters']['Photon Energy [eV]'])
        p1 = float(default_ini['Parameters']['grazing angle [mrad]'])
        p2 = int(default_ini['Parameters']['n profiles'])
        p3 = int(default_ini['Parameters']['filter width'])
        p4 = float(default_ini['Parameters']['projection From Divergence'])

        if easyqt.get_yes_or_no('Load new files?\n' +
                                '[ESC load file(s) of previous run]'):

            fnameH = easyqt.get_file_names(title='Select DPC Horizontal\n' +
                                           '(and Vertical if you want)')
            fnameV = None

            if len(fnameH) == 1:
                fnameH = fnameH[0]
                wpu.print_blue('MESSAGE: Horiz DPC: Loading ' + fnameH)
            elif len(fnameH) == 0:
                fnameH = None
            elif len(fnameH) == 2:
                [fnameH, fnameV] = fnameH
                wpu.print_blue('MESSAGE: Horiz DPC: Loading ' + fnameH)
                wpu.print_blue('MESSAGE: Vert DPC: Loading ' + fnameV)

            if fnameV is None:
                fnameV = easyqt.get_file_names(title='Select DPC Vertical')

                if len(fnameV) == 1:
                    fnameV = fnameV[0]
                    wpu.print_blue('MESSAGE: Vert DPC: Loading ' + fnameV)

                elif len(fnameV) == 0:
                    fnameV = None

        else:
            fnameH = default_ini['Files']['dpc H']
            fnameV = default_ini['Files']['dpc V']

            wpu.print_blue('MESSAGE: Horiz DPC: Loading ' + fnameH)
            wpu.print_blue('MESSAGE: Vert DPC: Loading ' + fnameV)

            if fnameH == 'None':
                fnameH = None
            if fnameV == 'None':
                fnameV = None

        phenergy = easyqt.get_float("Enter Photon Energy [KeV]",
                                    title='Experimental Values',
                                    default_value=p0*1e-3)*1e3

        grazing_angle = easyqt.get_float('Grazing angle [mrad]\n' +
                                         '[0.0 to ignore projection]',
                                         title='Experimental Values',
                                         default_value=p1)*1e-3

        projectionFromDiv = easyqt.get_float('projection From Divergence\n' +
                                             '[Multiplication factor]',
                                             title='Experimental Values',
                                             default_value=p4)

        nprofiles = easyqt.get_int("Number of profiles to plot",
                                   title='Experimental Values',
                                   default_value=p2)

        filter_width = easyqt.get_int("Width fo uniform filter [pixels]",
                                      title='Experimental Values',
                                      default_value=p3, max_=1e6)


        remove1stOrderDPC = easyqt.get_yes_or_no("Remove 1st Order DPC?",
                                                 title='Experimental Values')

        remove2ndOrder = easyqt.get_yes_or_no("Remove 2nd Order?",
                                              title='Experimental Values')

    elif len(sys_argv) == 10:

        if 'none' in sys_argv[1].lower():
            fnameH = None
        else:
            fnameH = sys_argv[1]

        if 'none' in sys_argv[2].lower():
            fnameV = None
        else:
            fnameV = sys_argv[2]

        phenergy = float(sys_argv[3])*1e3
        nprofiles = int(sys_argv[4])
        filter_width = int(sys_argv[5])
        grazing_angle = float(sys_argv[6])*1e-3
        projectionFromDiv = float(sys_argv[7])
        remove1stOrderDPC = bool(int(argv[8]))
        remove2ndOrder = bool(int(argv[9]))

    else:

        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n' +
              os.path.basename(__file__) + ' : (no inputs) load dialogs \n'
              '\n' +
              os.path.basename(__file__) + ' [args] \n'
              '\n'
              'arg1: file name DPC Horiz (type "None" '
              '      to ignore it)\n'
              'arg2: file name DPC Vert (type "None" '
              '      to ignore it)\n'
              'arg3: Photon Energy [KeV]\n'
              'arg4: Number of profiles to plot\n'
              'arg5: Width  of uniform filter [pixels]\n'
              'arg6: Grazing angle to project coordinates to mirror [mrad], use zero to ignore\n'
              'arg7: Projection From Divergence, use 1 to ignore'
              '\n')

        exit(-1)

    wpu.set_at_ini_file(inifname, 'Files', 'DPC H', fnameH)
    wpu.set_at_ini_file(inifname, 'Files', 'DPC V', fnameV)
    wpu.set_at_ini_file(inifname, 'Parameters', 'Photon Energy [eV]', phenergy)
    wpu.set_at_ini_file(inifname, 'Parameters',
                        'grazing angle [mrad]', grazing_angle*1e3)
    wpu.set_at_ini_file(inifname, 'Parameters',
                        'projection From Divergence', projectionFromDiv)
    wpu.set_at_ini_file(inifname, 'Parameters', 'n profiles', nprofiles)
    wpu.set_at_ini_file(inifname, 'Parameters', 'filter width', filter_width)

    wpu.set_at_ini_file(inifname, 'Parameters', 'Remove 1st Order DPC', remove1stOrderDPC)
    wpu.set_at_ini_file(inifname, 'Parameters', 'Remove 2nd Order', remove2ndOrder)

    return (fnameH, fnameV,
            phenergy, grazing_angle, projectionFromDiv,
            nprofiles, remove1stOrderDPC, remove2ndOrder, filter_width)


# %% Main functions to be used from the outside

def dpc_profile_analysis(fnameH, fnameV,
                         phenergy,
                         grazing_angle=0.0, projectionFromDiv=1.0,
                         nprofiles=1,
                         remove1stOrderDPC=False,
                         remove2ndOrder=False,
                         filter_width=0):

    wavelength = wpu.hc/phenergy

    if fnameH is not None:
        diffPhaseH, virtual_pixelsize, _ = wpu.load_sdf_file(fnameH)

    if fnameV is not None:
        diffPhaseV, virtual_pixelsize, _ = wpu.load_sdf_file(fnameV)

    if fnameH is None:
        diffPhaseH = diffPhaseV*np.nan

    if fnameV is None:
        diffPhaseV = diffPhaseH*np.nan
        saveFileSuf = fnameH.rsplit('/', 1)[0] + '/profiles/' +\
                      fnameH.rsplit('/', 1)[1]
        saveFileSuf = saveFileSuf.rsplit('_X')[0] + '_profiles'
    else:
        saveFileSuf = fnameV.rsplit('/', 1)[0] + '/profiles/' +\
                      fnameV.rsplit('/', 1)[1]
        saveFileSuf = saveFileSuf.rsplit('_Y')[0] + '_profiles'

    if not os.path.exists(saveFileSuf.rsplit('/', 1)[0]):
        os.makedirs(saveFileSuf.rsplit('/', 1)[0])


    (dataH, dataV,
     labels_H, labels_V,
     fit_coefs) = _n_profiles_H_V(diffPhaseH,
                                  diffPhaseV,
                                  virtual_pixelsize,
                                  'DPC [rad/m]',
                                  titleH='WF DPC Horz',
                                  titleV='WF DPC Vert',
                                  saveFileSuf=saveFileSuf,
                                  nprofiles=nprofiles,
                                  remove1stOrderDPC=remove1stOrderDPC,
                                  filter_width=filter_width)

    fit_coefsH = np.array(fit_coefs[0])
    fit_coefsV = np.array(fit_coefs[1])

    print(fit_coefsH)
    print(fit_coefsV)

    if __name__ == '__main__':
        wpu.log_this(preffname=saveFileSuf, inifname=inifname)

    if fnameH is not None:

        radii_fit_H = (2*np.pi/wavelength/fit_coefsH[:][0])

        wpu.print_blue('MESSAGE: Radius H from fit profiles: ')
        print(radii_fit_H)
        wpu.log_this('radius fit Hor = ' + str(radii_fit_H))

        integratedH = integrate_DPC_cumsum(dataH, wavelength,
                                           #grazing_angle=grazing_angle,
                                           remove2ndOrder=remove2ndOrder,
                                           xlabel='x',
                                           labels=labels_H,
                                           titleStr='Horizontal, ',
                                           saveFileSuf=saveFileSuf + '_X')

        curv_H = curv_from_height(integratedH, virtual_pixelsize[0],
                                  #grazing_angle=grazing_angle,
                                  #projectionFromDiv=projectionFromDiv,
                                  xlabel='x',
                                  labels=labels_H,
                                  titleStr='Horizontal, ',
                                  saveFileSuf=saveFileSuf + '_X')

    if fnameV is not None:

        radii_fit_V = (2*np.pi/wavelength/fit_coefsV[:][0])

        wpu.print_blue('MESSAGE: Radius V from fit profiles: ')
        print(radii_fit_V)
        wpu.log_this('radius fit Vert = ' + str(radii_fit_V))

        integratedV = integrate_DPC_cumsum(dataV, wavelength,
                                           grazing_angle=grazing_angle,
                                           projectionFromDiv=projectionFromDiv,
                                           remove2ndOrder=remove2ndOrder,
                                           xlabel='y',
                                           labels=labels_V,
                                           titleStr='Vertical, ',
                                           saveFileSuf=saveFileSuf + '_Y')

        curv_V = curv_from_height(integratedV, virtual_pixelsize[1],
                                  grazing_angle=grazing_angle,
                                  projectionFromDiv=projectionFromDiv,
                                  xlabel='y',
                                  labels=labels_V,
                                  titleStr='Vertical, ',
                                  saveFileSuf=saveFileSuf + '_Y')

def dpc_profile_analysis_modi(fnameH, fnameV,
                         phenergy,
                         grazing_angle=0.0, projectionFromDiv=1.0,
                         nprofiles=1,
                         remove1stOrderDPC=False,
                         remove2ndOrder=False,
                         filter_width=0):

    wavelength = wpu.hc/phenergy

    if fnameH is not None:
        diffPhaseH, virtual_pixelsize, _ = wpu.load_sdf_file(fnameH)

    if fnameV is not None:
        diffPhaseV, virtual_pixelsize, _ = wpu.load_sdf_file(fnameV)

    if fnameH is None:
        diffPhaseH = diffPhaseV*np.nan

    if fnameV is None:
        diffPhaseV = diffPhaseH*np.nan
        saveFileSuf = fnameH.rsplit('/', 1)[0] + '/profiles/' +\
                      fnameH.rsplit('/', 1)[1]
        saveFileSuf = saveFileSuf.rsplit('_X')[0] + '_profiles'
    else:
        saveFileSuf = fnameV.rsplit('/', 1)[0] + '/profiles/' +\
                      fnameV.rsplit('/', 1)[1]
        saveFileSuf = saveFileSuf.rsplit('_Y')[0] + '_profiles'

    if not os.path.exists(saveFileSuf.rsplit('/', 1)[0]):
        os.makedirs(saveFileSuf.rsplit('/', 1)[0])

    '''
        force the dpc_profile to smaller range.
    '''
    from matplotlib import  pyplot as plt
    #
    # print(diffPhaseV.shape)
    plt.figure()
    plt.imshow(diffPhaseV)
    plt.show()

    diffPhaseV = diffPhaseV[71:461, :]
    diffPhaseH = diffPhaseH[71:461, :]
    #
    plt.figure()
    plt.imshow(diffPhaseV)
    plt.show()

    (dataH, dataV,
     labels_H, labels_V,
     fit_coefs) = _n_profiles_H_V(diffPhaseH,
                                  diffPhaseV,
                                  virtual_pixelsize,
                                  'DPC [rad/m]',
                                  titleH='WF DPC Horz',
                                  titleV='WF DPC Vert',
                                  saveFileSuf=saveFileSuf,
                                  nprofiles=nprofiles,
                                  remove1stOrderDPC=remove1stOrderDPC,
                                  filter_width=filter_width)

    fit_coefsH = np.array(fit_coefs[0])
    fit_coefsV = np.array(fit_coefs[1])

    print(fit_coefsH)
    print(fit_coefsV)

    if __name__ == '__main__':
        wpu.log_this(preffname=saveFileSuf, inifname=inifname)

    if fnameH is not None:

        radii_fit_H = (2*np.pi/wavelength/fit_coefsH[:][0])

        wpu.print_blue('MESSAGE: Radius H from fit profiles: ')
        print(radii_fit_H)
        wpu.log_this('radius fit Hor = ' + str(radii_fit_H))

        integratedH = integrate_DPC_cumsum(dataH, wavelength,
                                           #grazing_angle=grazing_angle,
                                           remove2ndOrder=remove2ndOrder,
                                           xlabel='x',
                                           labels=labels_H,
                                           titleStr='Horizontal, ',
                                           saveFileSuf=saveFileSuf + '_X')

        curv_H = curv_from_height(integratedH, virtual_pixelsize[0],
                                  #grazing_angle=grazing_angle,
                                  #projectionFromDiv=projectionFromDiv,
                                  xlabel='x',
                                  labels=labels_H,
                                  titleStr='Horizontal, ',
                                  saveFileSuf=saveFileSuf + '_X')

    if fnameV is not None:

        radii_fit_V = (2*np.pi/wavelength/fit_coefsV[:][0])

        wpu.print_blue('MESSAGE: Radius V from fit profiles: ')
        print(radii_fit_V)
        wpu.log_this('radius fit Vert = ' + str(radii_fit_V))

        integratedV = integrate_DPC_cumsum(dataV, wavelength,
                                           grazing_angle=grazing_angle,
                                           projectionFromDiv=projectionFromDiv,
                                           remove2ndOrder=remove2ndOrder,
                                           xlabel='y',
                                           labels=labels_V,
                                           titleStr='Vertical, ',
                                           saveFileSuf=saveFileSuf + '_Y')

        curv_V = curv_from_height(integratedV, virtual_pixelsize[1],
                                  grazing_angle=grazing_angle,
                                  projectionFromDiv=projectionFromDiv,
                                  xlabel='y',
                                  labels=labels_V,
                                  titleStr='Vertical, ',
                                  saveFileSuf=saveFileSuf + '_Y')


if __name__ == '__main__':

    (fnameH, fnameV,
     phenergy, grazing_angle, projectionFromDiv,
     nprofiles, remove1stOrderDPC, remove2ndOrder, filter_width) = _intial_gui_setup(sys.argv)

    dpc_profile_analysis(fnameH, fnameV,
                         phenergy, grazing_angle,
                         projectionFromDiv, nprofiles, remove1stOrderDPC, remove2ndOrder, filter_width)




