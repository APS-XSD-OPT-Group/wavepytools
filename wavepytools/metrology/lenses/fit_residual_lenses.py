#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
# =============================================================================
# %%
# =============================================================================
import sys

import numpy as np

from numpy.fft import fft2, ifft2, fftfreq

if len(sys.argv) > 1 or False:
    import matplotlib
    matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import os

import wavepy.utils as wpu

wpu._mpl_settings_4_nice_graphs()

from scipy.optimize import curve_fit, minimize

import pickle
from wavepy.utils import easyqt
import itertools


global inifname  # name of .ini file
inifname = os.curdir + '/.' + os.path.basename(__file__).replace('.py', '.ini')
# =============================================================================
# %% Plot dpx and dpy and fit Curvature Radius of WF
# =============================================================================


def fit_radius_dpc(dpx, dpy, pixelsize, radius4fit, kwave,
                   saveFigFlag=False, str4title=''):

    xVec = wpu.realcoordvec(dpx.shape[1], pixelsize[1])
    yVec = wpu.realcoordvec(dpx.shape[0], pixelsize[0])

    lim_x = np.argwhere(xVec >= -radius4fit*1.01)[0, 0]
    lim_y = np.argwhere(yVec >= -radius4fit*1.01)[0, 0]

    xmatrix, ymatrix = np.meshgrid(xVec[lim_x:-lim_x+1], yVec[lim_y:-lim_y+1])

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(str4title + 'Phase [rad]', fontsize=14)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    ax1.plot(xVec[lim_x:-lim_x+1]*1e6, dpx[dpx.shape[1]//4, lim_x:-lim_x+1],
             '-ob', label='1/4')
    ax1.plot(xVec[lim_x:-lim_x+1]*1e6, dpx[dpx.shape[1]//2, lim_x:-lim_x+1],
             '-or', label='1/2')
    ax1.plot(xVec[lim_x:-lim_x+1]*1e6, dpx[dpx.shape[1]//4*3, lim_x:-lim_x+1],
             '-og', label='3/4')

    lin_fitx = np.polyfit(xVec[lim_x:-lim_x+1],
                          dpx[dpx.shape[1]//2, lim_x:-lim_x+1], 1)
    lin_funcx = np.poly1d(lin_fitx)
    ax1.plot(xVec[lim_x:-lim_x+1]*1e6, lin_funcx(xVec[lim_x:-lim_x+1]),
             '--c', lw=2,
             label='Fit 1/2')
    curvrad_x = kwave/(lin_fitx[0])

    wpu.print_blue('lin_fitx[0] x: {:.3g} m'.format(lin_fitx[0]))
    wpu.print_blue('lin_fitx[1] x: {:.3g} m'.format(lin_fitx[1]))

    wpu.print_blue('Curvature Radius of WF x: {:.3g} m'.format(curvrad_x))

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax1.set_xlabel(r'[$\mu m$]')
    ax1.set_ylabel('dpx [radians]')
    ax1.legend(loc=0, fontsize='small')
    ax1.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_x),
                  fontsize=16)
    ax1.set_adjustable('box-forced')

    ax2.plot(yVec[lim_y:-lim_y+1]*1e6, dpy[lim_y:-lim_y+1, dpy.shape[0]//4],
             '-ob', label='1/4')
    ax2.plot(yVec[lim_y:-lim_y+1]*1e6, dpy[lim_y:-lim_y+1, dpy.shape[0]//2],
             '-or', label='1/2')
    ax2.plot(yVec[lim_y:-lim_y+1]*1e6, dpy[lim_y:-lim_y+1, dpy.shape[0]//4*3],
             '-og', label='3/4')

    lin_fity = np.polyfit(yVec[lim_y:-lim_y+1],
                          dpy[lim_y:-lim_y+1, dpy.shape[0]//2], 1)
    lin_funcy = np.poly1d(lin_fity)
    ax2.plot(yVec[lim_y:-lim_y+1]*1e6, lin_funcy(yVec[lim_y:-lim_y+1]),
             '--c', lw=2,
             label='Fit 1/2')
    curvrad_y = kwave/(lin_fity[0])
    wpu.print_blue('Curvature Radius of WF y: {:.3g} m'.format(curvrad_y))

    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax2.set_xlabel(r'[$\mu m$]')
    ax2.set_ylabel('dpy [radians]')
    ax2.legend(loc=0, fontsize='small')
    ax2.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_y),
                  fontsize=16)
    ax2.set_adjustable('box-forced')

    if saveFigFlag:
        wpu.save_figs_with_idx(fname2save, extension='png')
    plt.show(block=False)


# %% center fig

def center_CM_2darray(array):
    '''
    crop the array in order to have the max at the center of the array
    '''

    array = np.copy(array)

    array[np.isnan(array)] = 0.0

    #    min_array = np.min(array)
    #    array -= min_array

    i, j = np.mgrid[0:array.shape[0], 0:array.shape[1]]

    center_i = int(np.average(i, weights=array))
    center_j = int(np.average(j, weights=array))

    if 2*center_i > array.shape[0]:
        array = array[2*center_i-array.shape[0]:, :]
    else:
        array = array[0:2*center_i, :]

    if 2*center_j > array.shape[1]:
        array = array[:, 2*center_j-array.shape[1]:]
    else:
        array = array[:, 0:2*center_j]

    #    array += min_array

    return array


def center_lens_array_max(array, pixelSize, radius):
    '''
    crop the array in order to have the max at the center of the array
    '''

    array = np.copy(array)

    xx, yy = wpu.grid_coord(array, pixelSize)
    r2 = np.sqrt(xx**2 + yy**2)

    mask = 0*xx
    mask[np.where((r2 < radius*1.1) & (r2 > radius*.9))] = 1.0

    mean_at_r2 = np.mean(array*mask)
    mask2 = 0*xx
    mask2[np.where(array > mean_at_r2)] = 1.0

    i, j = np.mgrid[0:array.shape[0], 0:array.shape[1]]

    center_i = int(np.average(i, weights=array*mask2))
    center_j = int(np.average(j, weights=array*mask2))

    if 2*center_i > array.shape[0]:
        array = array[2*center_i-array.shape[0]:, :]
    else:
        array = array[0:2*center_i, :]

    if 2*center_j > array.shape[1]:
        array = array[:, 2*center_j-array.shape[1]:]
    else:
        array = array[:, 0:2*center_j]

    return array


def _biggest_radius(array, pixelSize, radius4fit):

    bool_x = (array.shape[0] // 2 < radius4fit//pixelSize[0])
    bool_y = (array.shape[1] // 2 < radius4fit//pixelSize[1])

    if bool_x or bool_y:
        radius4fit = .9*np.min((thickness.shape[0]*pixelSize[0] / 2,
                                thickness.shape[1]*pixelSize[1] / 2))

        wpu.print_red('WARNING: Image size smaller than ' +
                      'the region for fit')

        wpu.print_red('WARNING: New Radius:' +
                      ' {:.3f}um'.format(radius4fit*1e6))

    return radius4fit


# %%
def center_lens_array_max_fit(array, pixelSize, radius4fit=100e-6):
    '''
    crop the array in order to have the max at the center of the array. It uses
    a fitting procedure of a 2D parabolic function to determine the center

    '''

    radius4fit = _biggest_radius(array, pixelSize, radius4fit*.8)

    array = np.copy(array)

    xx, yy = wpu.grid_coord(array, pixelSize)

    (_, _,
     fitParameters) = fit_parabolic_lens_2d(array, pixelSize,
                                            radius4fit=radius4fit)

    center_i = np.argmin(np.abs(yy[:, 0]-fitParameters[2]))
    center_j = np.argmin(np.abs(xx[0, :]-fitParameters[1]))

    if 2*center_i > array.shape[0]:
        array = array[2*center_i-array.shape[0]:, :]
    else:
        array = array[0:2*center_i, :]

    if 2*center_j > array.shape[1]:
        array = array[:, 2*center_j-array.shape[1]:]
    else:
        array = array[:, 0:2*center_j]

    return array


# =============================================================================
# %% 1D Fit
# =============================================================================
def _lsq_fit_1D_parabola(yy, pixelsize):

    xx = wpu.realcoordvec(yy.shape[0], pixelsize)

    if np.all(np.isfinite(yy)):  # if there is no nan
        f = yy.flatten()
        x = xx.flatten()
    else:
        argNotNAN = np.isfinite(yy)
        f = yy[argNotNAN].flatten()
        x = xx[argNotNAN].flatten()

    X_matrix = np.vstack([x**2, x, x*0.0 + 1]).T

    beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

    fit = (beta_matrix[0]*xx**2 +
           beta_matrix[1]*xx +
           beta_matrix[2])

    if np.all(np.isfinite(yy)):
        mask = yy*0.0 + 1.0
    else:
        mask = yy*0.0 + 1.0
        mask[~argNotNAN] = np.nan

    return fit*mask, beta_matrix


# %%
def plot_residual_1d(xvec, data, fitted, str4title='',
                     saveFigFlag=False, saveAsciiFlag=False):

    # Plot Horizontal profile

    errorThickness = -data + fitted
    argNotNAN = np.isfinite(errorThickness)

    factorx, unitx = wpu.choose_unit(xvec)
    factory1, unity1 = wpu.choose_unit(data)
    factory2, unity2 = wpu.choose_unit(errorThickness)

    ptp = np.ptp(errorThickness[argNotNAN].flatten()*factory2)
    wpu.print_red('PV: {0:4.3g} '.format(ptp) + unity2[-1] + 'm')

    sigmaError = np.std(errorThickness[argNotNAN].flatten()*factory2)
    wpu.print_red('SDV: {0:4.3g} '.format(sigmaError) + unity2[-1] + 'm')

    str4title += '\n' + \
                 r'PV $= {0:.2f}$ '.format(ptp) + '$' + unity2 + '  m$, '\
                 'SDV $= {0:.2f}$ '.format(sigmaError) + '$' + unity2 + '  m$'

    plt.figure(figsize=(10, 7))
    ax1 = plt.gca()
    ax1.plot(xvec[argNotNAN]*factorx,
             data[argNotNAN]*factory1,
             '-ko', markersize=5, label='1D data')

    ax1.plot(xvec[argNotNAN]*factorx, fitted[argNotNAN]*factory1,
             '-+r', label='Fit parabolic')

    ax2 = ax1.twinx()

    # trick to add both axes to legend
    ax2.plot(np.nan, '-ko', label='1D data')
    ax2.plot(np.nan, '-+r', label='Fit parabolic')

    ax2.plot(xvec[argNotNAN]*factorx,
             errorThickness[argNotNAN]*factory2,
             '-+', markersize=5, label='fit residual')

    plt.title(str4title)

    for tl in ax2.get_yticklabels():
        tl.set_color('b')

    ax2.legend(loc=1, fontsize='small')
    # trick to add both axes to legend

    ax1.grid(color='gray')

    ax1.set_xlabel(r'[$' + unitx + ' m$]')
    ax1.set_ylabel(r'Thickness ' + r'[$' + unity1 + ' m$]')

    #    ax2.set_ylim([-20, 20])

    ax2.set_ylim(-1.1*np.max(np.abs(errorThickness[argNotNAN])*factory2),
                 1.1*np.max(np.abs(errorThickness[argNotNAN])*factory2))
    ax2.set_ylabel(r'Residual' + r'[$' + unity2 + ' m$]')
    ax2.grid(b='off')
    plt.xlim(-1.1*np.max(xvec*factorx), 1.1*np.max(xvec*factorx))

    plt.tight_layout(rect=(0, 0, 1, .98))

    if saveFigFlag:
        wpu.save_figs_with_idx(fname2save, extension='png')

    if saveAsciiFlag:
        csv_fname = wpu.get_unique_filename(fname2save, 'csv')
        np.savetxt(csv_fname,
                   np.transpose([xvec, data, fitted, fitted-data]),
                   delimiter=',\t',
                   header="xvec, data, fitted, residual, " + str4title,
                   fmt='%.6g')
    plt.show(block=False)

# =============================================================================
# %% 2D Fit
# =============================================================================


def _2Dparabol_4_fit(xy, Radius, xo, yo, offset):

    x, y = xy
    return (x - xo)**2/2/Radius + (y - yo)**2/2/Radius + offset


def _2Drotated_parabol_4_fit(xy, Radius_y, Radius_x, xo, yo, offset, theta):
    # never tested nor used

    x, y = xy

    x2 = (x-xo)*np.cos(theta*np.deg2rad(1)) + \
         (y-yo)*np.sin(theta*np.deg2rad(1))
    y2 = (x-xo)*np.cos(theta*np.deg2rad(1)) + \
         (y-yo)*np.sin(theta*np.deg2rad(1))

    return x2**2/2/Radius_x + y2**2/2/Radius_y + offset


def _lsq_fit_parabola(zz, pixelsize, mode='2D'):

    xx, yy = wpu.grid_coord(zz, pixelsize)

    if np.all(np.isfinite(zz)):  # if there is no nan
        f = zz.flatten()
        x = xx.flatten()
        y = yy.flatten()
    else:
        argNotNAN = np.isfinite(zz)
        f = zz[argNotNAN].flatten()
        x = xx[argNotNAN].flatten()
        y = yy[argNotNAN].flatten()

    if '2D' in mode:
        X_matrix = np.vstack([x**2 + y**2, x, y, x*0.0 + 1]).T

        beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

        fit = (beta_matrix[0]*(xx**2 + yy**2) +
               beta_matrix[1]*xx +
               beta_matrix[2]*yy +
               beta_matrix[3])

    elif '1Dx' in mode:
        X_matrix = np.vstack([x**2, x, y, x*0.0 + 1]).T

        beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

        fit = (beta_matrix[0]*(xx**2) +
               beta_matrix[1]*xx +
               beta_matrix[2]*yy +
               beta_matrix[3])

    elif '1Dy' in mode:
        X_matrix = np.vstack([y**2, x, y, x*0.0 + 1]).T

        beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

        fit = (beta_matrix[0]*(yy**2) +
               beta_matrix[1]*xx +
               beta_matrix[2]*yy +
               beta_matrix[3])

    if np.all(np.isfinite(zz)):
        mask = zz*0.0 + 1.0
    else:
        mask = zz*0.0 + 1.0
        mask[~argNotNAN] = np.nan

    R_o = 1/2/beta_matrix[0]
    x_o = -beta_matrix[1]/beta_matrix[0]/2
    y_o = -beta_matrix[2]/beta_matrix[0]/2
    offset = beta_matrix[3]

    popt = [R_o, x_o, y_o, offset]

    return fit*mask, popt


def fit_parabolic_lens_2d(thickness, pixelsize,
                          radius4fit, mode='2D'):

    # FIT
    xx, yy = wpu.grid_coord(thickness, pixelSize)
    mask = xx*np.nan

    lim_x = np.argwhere(xx[0, :] <= -radius4fit*1.01)[-1, 0]
    lim_y = np.argwhere(yy[:, 0] <= -radius4fit*1.01)[-1, 0]

    if '2D' in mode:

        r2 = np.sqrt(xx**2 + yy**2)
        mask[np.where(r2 < radius4fit)] = 1.0

    elif '1Dx' in mode:
        mask[np.where(xx**2 < radius4fit)] = 1.0
        lim_y = 2

    elif '1Dy' in mode:
        mask[np.where(yy**2 < radius4fit)] = 1.0
        lim_x = 2

    fitted, popt = _lsq_fit_parabola(thickness*mask, pixelSize, mode=mode)

    wpu.print_blue("Parabolic 2D Fit")
    wpu.print_blue("Curv Radius, xo, yo, offset")
    wpu.print_blue(popt)

    wpu.print_blue("Parabolic 2D Fit: Radius of 1 face  / nfaces, " +
                   "x direction: {:.4g} um".format(popt[0]*1e6))

    if (lim_x <= 1 or lim_y <= 1):
        thickness_cropped = thickness*mask
        fitted_cropped = fitted*mask
    else:
        thickness_cropped = (thickness[lim_y:-lim_y+1, lim_x:-lim_x+1] *
                             mask[lim_y:-lim_y+1, lim_x:-lim_x+1])
        fitted_cropped = (fitted[lim_y:-lim_y+1, lim_x:-lim_x+1] *
                          mask[lim_y:-lim_y+1, lim_x:-lim_x+1])

    return (thickness_cropped, fitted_cropped, popt)


def fit_nominal_lens_2d(thickness, pixelsize,
                        radius4fit,
                        p0=[20e-6, 1.005e-6, -.005e-6, -.005e-6],
                        bounds=([10e-6, -2.05e-6, -2.05e-6, -2.05e-6],
                                [50e-6, 2.05e-6, 2.05e-6, 2.05e-6]),
                        saveFigFlag=False, str4title='', kwargs4fit={}):

    #    thickness = center_CM_2darray(thickness)

    xmatrix, ymatrix = wpu.grid_coord(thickness, pixelSize)
    r2 = np.sqrt(xmatrix**2 + ymatrix**2)
    args4fit = np.where(r2.flatten() < radius4fit)

    mask = xmatrix*np.nan
    mask[np.where(r2 < radius4fit)] = 1.0

    data2fit = thickness.flatten()[args4fit]

    xxfit = xmatrix.flatten()[args4fit]
    yyfit = ymatrix.flatten()[args4fit]

    xyfit = [xxfit, yyfit]

    # FIT

    popt, pcov = curve_fit(_2Dparabol_4_fit, xyfit, data2fit,
                           p0=p0, bounds=bounds, method='trf',
                           **kwargs4fit)

    wpu.print_blue("Nominal Parabolic 2D Fit")
    wpu.print_blue("Curv Radius, xo, yo, offset")
    wpu.print_blue(popt)

    wpu.print_blue("Nominal Parabolic 2D Fit: Radius of 1 face  / nfaces, " +
                   "x direction: {:.4g} um".format(popt[0]*1e6))

    lim_x = np.argwhere(xmatrix[0, :] <= -radius4fit*1.01)[-1, 0]
    lim_y = np.argwhere(ymatrix[:, 0] <= -radius4fit*1.01)[-1, 0]

    fitted = _2Dparabol_4_fit([xmatrix, ymatrix],
                              popt[0], popt[1],
                              popt[2], popt[3])

    if (lim_x <= 1 or lim_y <= 1):
        thickness_cropped = thickness*mask
        fitted_cropped = fitted*mask
    else:
        thickness_cropped = (thickness[lim_y:-lim_y+1, lim_x:-lim_x+1] *
                             mask[lim_y:-lim_y+1, lim_x:-lim_x+1])
        fitted_cropped = (fitted[lim_y:-lim_y+1, lim_x:-lim_x+1] *
                          mask[lim_y:-lim_y+1, lim_x:-lim_x+1])

    return (thickness_cropped, fitted_cropped, popt)


# %%
def plot_residual_parabolic_lens_2d(thickness, pixelsize,
                                    fitted, fitParameters,
                                    saveFigFlag=False, savePickle=False,
                                    str4title='', saveSdfData=False,
                                    vlimErrSigma=1,
                                    plotProfileFlag=True,
                                    plot3dFlag=True,
                                    makeAnimation=False):

    xmatrix, ymatrix = wpu.grid_coord(thickness, pixelsize)

    errorThickness = thickness - fitted
    argNotNAN = np.isfinite(errorThickness)

    factorx, unitx = wpu.choose_unit(xmatrix)
    factory, unity = wpu.choose_unit(ymatrix)
    factorz, unitz = wpu.choose_unit(errorThickness[argNotNAN])

    ptp = np.ptp(errorThickness[argNotNAN].flatten()*factorz)
    wpu.print_red('PV: {0:4.3g} '.format(ptp) + unitz[-1] + 'm')

    sigmaError = np.std(errorThickness[argNotNAN].flatten()*factorz)
    wpu.print_red('SDV: {0:4.3g} '.format(sigmaError) + unitz[-1] + 'm')

    str4title += r'Residual, ' + \
                 r'R $= {:.4g} \mu m$,'.format(fitParameters[0]*1e6) + '\n' + \
                 r'PV $= {0:.2f}$ '.format(ptp) + '$' + unitz + '  m$, '\
                 'SDV $= {0:.2f}$ '.format(sigmaError) + '$' + unitz + '  m$'

    # Plot Histogram

    plt.figure(figsize=(7, 8))
    plt.hist(errorThickness[argNotNAN]*factorz,
             100, color='r', histtype='step')
    plt.xlabel(r'Residual [$' + unitz + '  m$ ]')
    plt.title(str4title)

    if saveFigFlag:
        wpu.save_figs_with_idx(fname2save, extension='png')

    plt.show(block=False)

    # Plot Profiles

    vlimErr = wpu.mean_plus_n_sigma(errorThickness[argNotNAN]*factorz,
                                    vlimErrSigma/2)
    cmap4graph = plt.cm.Spectral_r
    cmap4graph.set_over('m')
    cmap4graph.set_under('c')

    if plotProfileFlag:

        wpu.plot_profile(xmatrix*factorx,
                         ymatrix*factory,
                         errorThickness*factorz,
                         title=str4title,
                         xlabel=r'[$' + unitx + '  m$ ]',
                         ylabel=r'[$' + unity + '  m$ ]',
                         zlabel=r'[$' + unitz + '  m$ ]',
                         arg4main={'cmap': 'Spectral_r',
                                   'vmin': -vlimErr,
                                   'vmax': vlimErr,
                                   'extend': 'both'})

    if savePickle or saveFigFlag:
        fig = plt.figure(figsize=(10, 7))

        cf = plt.contourf(xmatrix*factorx,
                          ymatrix*factory,
                          errorThickness*factorz, 256,
                          cmap=cmap4graph,
                          extend='both')

        plt.clim(-vlimErr, vlimErr)
        plt.contour(cf, levels=cf.levels[::32],
                    colors='gray')

        plt.xlabel(r'[$' + unitx + '  m$ ]', fontsize=22)
        plt.ylabel(r'[$' + unity + '  m$ ]', fontsize=22)
        plt.title(str4title, fontsize=22)
        cbar = plt.colorbar(cf, shrink=.8, aspect=20)
        #        cbar.set_clim(-vlimErr, vlimErr)
        cbar.ax.set_title(r'[$' + unitz + '  m$ ]', y=1.01)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(color='grey')

        if saveFigFlag:
            wpu.save_figs_with_idx(fname2save, extension='png')

        if savePickle:
            wpu.save_figs_with_idx_pickle(fig, fname2save)

        plt.show(block=True)

    # Plot 3D

    if plot3dFlag:

        wpu.print_red('MESSAGE: Ploting 3d in the background')

        fig = plt.figure(figsize=(10, 7), facecolor="white")
        ax = fig.gca(projection='3d')
        plt.tight_layout(pad=2.5)

        surf = ax.plot_trisurf(xmatrix[argNotNAN].flatten()*factorx,
                               ymatrix[argNotNAN].flatten()*factory,
                               errorThickness[argNotNAN].flatten()*factorz,
                               vmin=-vlimErr, vmax=vlimErr,
                               cmap=cmap4graph, linewidth=0.1, shade=False)

        ax.view_init(azim=-120, elev=40)

        plt.xlabel(r'$x$ [$' + unitx + '  m$ ]')
        plt.ylabel(r'$y$ [$' + unity + '  m$ ]')

        plt.title(str4title)

        cbar = plt.colorbar(surf, shrink=.8, aspect=20, extend='both')
        cbar.ax.set_title(r'[$' + unitz + '  m$ ]', y=1.01)

        plt.tight_layout()

        if saveFigFlag:
            wpu.save_figs_with_idx(fname2save, extension='png')

            ax.view_init(azim=690, elev=40)

            wpu.save_figs_with_idx(fname2save, extension='png')

        if makeAnimation:
            #            plt.show(block=False)
            plt.pause(1.0)
            wpu.rocking_3d_figure(ax,
                                  wpu.get_unique_filename(fname2save, 'gif'),
                                  elevOffset=45, azimOffset=60,
                                  elevAmp=0, azimAmpl=-1, dpi=80, npoints=5)

        plt.pause(1.0)
        plt.close('all')
        #    plt.show(block=True)

    if saveSdfData:
        mask_for_sdf = errorThickness*0.0
        mask_for_sdf[~argNotNAN] = 1.0
        errorThickness[~argNotNAN] = 00000000
        wpu.save_sdf_file(errorThickness, pixelsize,
                          wpu.get_unique_filename(fname2save +
                                                  '_residual', 'sdf'))
        wpu.save_sdf_file(mask_for_sdf, pixelsize,
                          wpu.get_unique_filename(fname2save +
                                                  '_residual_mask', 'sdf'))

    return sigmaError/factorz, ptp/factorz


# %%
def slope_error_hist(thickness_cropped, fitted, pixelSize,
                     delta=1, sourcedistance=1.0,
                     saveFigFlag=True, str4title=''):

    errorThickness = thickness_cropped-fitted

    plt.figure(figsize=(15, 8))
    plt.subplot(121)

    slope_error_h = np.diff(errorThickness, axis=0)/pixelSize[0]*delta
    argNotNAN = np.isfinite(slope_error_h)
    factor_seh, unit_seh = wpu.choose_unit(slope_error_h[argNotNAN])
    sigma_seh = np.std(slope_error_h[argNotNAN].flatten())

    plt.hist(slope_error_h[argNotNAN].flatten()*factor_seh,
             100, histtype='stepfilled')
    plt.xlabel(r'Slope Error [$  ' + unit_seh + ' rad$ ]')
    plt.title('Horizontal, SDV = ' +
              '{:.2f}'.format(sigma_seh*factor_seh) +
              ' $' + unit_seh + ' rad$')

    plt.subplot(122)

    slope_error_v = np.diff(errorThickness, axis=1)/pixelSize[1]*delta
    argNotNAN = np.isfinite(slope_error_v)
    factor_sev, unit_sev = wpu.choose_unit(slope_error_v[argNotNAN])
    sigma_sev = np.std(slope_error_v[argNotNAN].flatten())

    plt.hist(slope_error_v[argNotNAN].flatten()*factor_sev,
             100, histtype='stepfilled')
    plt.xlabel(r'Slope Error [$  ' + unit_sev + ' rad$ ]')
    plt.title('Vertical, SDV = ' +
              '{:.2f}'.format(sigma_sev*factor_sev) +
              ' $' + unit_sev + ' rad$')

    if delta!= 1:
        str4title += ' WF slope error'
    else:
        str4title += ' Thickness slope error'
    plt.suptitle(str4title, fontsize=18, weight='bold')

    if saveFigFlag:
        wpu.save_figs_with_idx(fname2save, extension='png')

    wpu.log_this('Slope Error Hor SDV = ' +
                 '{:.3f}'.format(sigma_seh*factor_seh) + unit_seh + ' rad')
    wpu.log_this('Slope Error Ver SDV = ' +
                 '{:.3f}'.format(sigma_sev*factor_sev) + unit_sev + ' rad')

    plt.show(block=True)

    return sigma_seh, sigma_sev


# %%
def load_pickle_surf(fname, plotSurf=True):

    file = open(fname, 'rb')
    figx = pickle.load(file)
    plt.show(block=True)

    data = figx.axes[0].images[0].get_array().data
    [xi, xf, yi, yf] = figx.axes[0].images[0].get_extent()

    ax = figx.axes[0].images[0].get_axes()

    title = figx.axes[0].images[0].axes.properties()['title']
    xlabel = figx.axes[0].images[0].axes.properties()['xlabel']
    ylabel = figx.axes[0].images[0].axes.properties()['ylabel']
    cmap = figx.axes[0].images[0].properties()['cmap'].name

    xxGrid, yyGrid = np.meshgrid(np.linspace(xi, xf, data.shape[1]),
                                 np.linspace(yi, yf, data.shape[0]),
                                 indexing='xy')

    if plotSurf:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xxGrid, yyGrid,  data,
                               rstride=data.shape[0]//101+1,
                               cstride=data.shape[1]//101+1,
                               cmap=cmap, linewidth=0.1)

        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=22)

        plt.title(title, fontsize=28, weight='bold')
        plt.colorbar(surf, shrink=.8, aspect=20)

        plt.tight_layout()
        plt.show(block=False)

    return data, xxGrid, yyGrid


def _intial_gui_setup():

    #    global inifname  # name of .ini file
    #    pwd, inifname = argvzero.rsplit('/', 1)
    #    inifname = pwd + '/.' + inifname.replace('.py', '.ini')

    defaults = wpu.load_ini_file(inifname)

    fname = easyqt.get_file_names("File to Plot (Pickle or sdf)")

    if fname == []:
        fname = defaults['Files'].get('file with thickness')

    else:
        fname = fname[0]
        defaults['Files']['file with thickness'] = fname

    wpu.print_blue('MESSAGE: Loading File: ' + fname)

    with open(inifname, 'w') as configfile:
            defaults.write(configfile)

    if defaults is None:
        p1 = ''
        p2 = 100e-6

    else:
        p1 = defaults['Parameters'].get('String for Titles')
        p2 = float(defaults['Parameters'].get('Nominal Radius For Fitting'))
        p3 = defaults['Parameters']['Diameter of active area for fitting']
        p4 = defaults['Parameters']['Lens Geometry']

    str4title = easyqt.get_string('String for Titles',
                                  'Enter String', p1)

    if str4title != '':
        defaults['Parameters']['String for Titles'] = str4title

    wpu.print_blue('MESSAGE: Loading Pickle: ' + fname)

    if p4 == '2D Lens Stigmatic Lens':
        menu_choices4lensmode = ['2D Lens Stigmatic Lens',
                                 '1Dx Horizontal focusing',
                                 '1Dy Vertical focusing']
    elif p4 == '1Dx Horizontal focusing':
        menu_choices4lensmode = ['1Dx Horizontal focusing',
                                 '1Dy Vertical focusing',
                                 '2D Lens Stigmatic Lens']
    elif p4 == '1Dy Vertical focusing':
        menu_choices4lensmode = ['1Dy Vertical focusing',
                                 '1Dx Horizontal focusing',
                                 '2D Lens Stigmatic Lens']

    lensGeometry = easyqt.get_choice(message='Lens Geometry',
                                     title='Input Parameter',
                                     choices=menu_choices4lensmode)

    nominalRadius = easyqt.get_float('Nominal Curvature Radius' +
                                     ' for Fitting [um]',
                                     title='Input Parameter',
                                     default_value=p2*1e6)*1e-6

    diameter4fit_str = easyqt.get_string('Diameter of active area for fitting [um]',
                                         title='Input Parameter',
                                         default_response=p3)

    diameter4fit_list = [float(a)*1e-6 for a in diameter4fit_str.split(',')]

    defaults['Parameters']['Nominal Radius For Fitting'] = str(nominalRadius)
    defaults['Parameters']['Diameter of active area for fitting'] = diameter4fit_str
    defaults['Parameters']['Lens Geometry'] = lensGeometry

    with open(inifname, 'w') as configfile:
            defaults.write(configfile)

    return fname, str4title, nominalRadius, diameter4fit_list, lensGeometry


def _load_experimental_pars(argv):

    global gui_mode

    if len(argv) == 6:

        fname = argv[1]
        str4title = argv[2]
        nominalRadius = float(argv[3])*1e-6

        menu_choices4lensmode = ['2D Lens Stigmatic Lens',
                                 '1Dx Horizontal focusing',
                                 '1Dy Vertical focusing']
        lensGeometry = menu_choices4lensmode[int(argv[4])]

        diameter4fit_list = [float(a)*1e-6 for a in argv[5].split(',')]

        for i, arg in enumerate(argv):
            print('arg {}: '.format(i) + argv[i])

        gui_mode = False
        wpu.print_red(argv)

    elif len(argv) == 1:

        (fname, str4title,
         nominalRadius, diameter4fit_list,
         lensGeometry) = _intial_gui_setup()

        gui_mode = True

    else:
        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n'
              'fit_residual_lenses.py : (no inputs) load dialogs \n'
              '\n'
              'fit_residual_lenses.py [args] \n'
              '\n'
              'arg1: file name with thickness image\n'
              'arg2: String for Titles\n'
              'arg3: nominal curvature radius for fitting\n'
              'arg4: index for lens geometry:\n'
              '\t0 : 2D Lens Stigmatic Lens\n'
              '\t1 : 1Dx Horizontal focusing\n'
              '\t2 : 1Dy Vertical focusing\n'
              'arg5: diameter4fit_list:\n'
              '\n')

        for i, arg in enumerate(argv):
            print('arg {}: '.format(i) + argv[i])

        exit(-1)

    if str4title != '':
        str4title += ', '

    return fname, str4title, nominalRadius, diameter4fit_list, lensGeometry

# =============================================================================
# %% Main
# =============================================================================

if __name__ == '__main__':

    # %% initial pars

    (fname, str4title, nominalRadius,
     diameter4fit_list, lensGeometry) = _load_experimental_pars(sys.argv)

    fname2save = fname.split('.')[0].split('/')[-1] + '_fit'

    data_dir = fname.rsplit('/', 1)[0]
    print(fname)
    print(data_dir)
    os.chdir(data_dir)
    os.makedirs('residuals', exist_ok=True)
    os.chdir('residuals')

    wpu.log_this(preffname=fname.split('.')[0].split('/')[-1] + '_fit',
                 inifname=inifname)

    # %% Load Input File

    if fname.split('.')[-1] == 'sdf':
        thickness, pixelSize, headerdic = wpu.load_sdf_file(fname)
        xx, yy = wpu.realcoordmatrix(thickness.shape[1], pixelSize[1],
                                     thickness.shape[0], pixelSize[0])

    elif fname.split('.')[-1] == 'pickle':

            thickness, xx, yy = load_pickle_surf(fname, False)

            thickness *= 1e-6
            #            thickness *= -1.0 # minus1 here
            xx *= 1e-6
            yy *= 1e-6
            pixelSize = [np.mean(np.diff(xx[0, :])),
                         np.mean(np.diff(yy[:, 0]))]

    else:
        wpu.print_red('ERROR: Wrong file type!')
        exit(-1)

    thickness -= np.nanmin(thickness)
    saveFigFlag = True

    # %% Crop

    metrology_flag = False
    if metrology_flag:
        thickness_temp = np.copy(thickness)

        thickness_temp[np.isnan(thickness)] = 0.0

        idx4crop = wpu.graphical_roi_idx(thickness_temp*1e6, verbose=True)

        thickness = wpu.crop_matrix_at_indexes(thickness, idx4crop)

        xx = wpu.crop_matrix_at_indexes(xx, idx4crop)
        yy = wpu.crop_matrix_at_indexes(yy, idx4crop)

        stride = thickness.shape[0] // 125

        if gui_mode:

            wpu.plot_profile(xx[::stride, ::stride]*1e6,
                             yy[::stride, ::stride]*1e6,
                             thickness[::stride, ::stride]*1e6,
                             xlabel=r'$x$ [$\mu m$ ]',
                             ylabel=r'$y$ [$\mu m$ ]',
                             zlabel=r'$z$ [$\mu m$ ]',
                             arg4main={'cmap': 'Spectral_r'})

    # %% Center image

    radius4centering = np.min(thickness.shape)*np.min(pixelSize)*.75
    #    thickness = -1*thickness + np.max(thickness)
    thickness = center_lens_array_max_fit(thickness, pixelSize,
                                          radius4centering)

    wpu.log_this('Array cropped to have the max at the center of the array',
                 preffname=fname2save)

    #    thickness = center_lens_array_max_fit(thickness, pixelSize, radius4centering)
    #    thickness = center_lens_array_max(thickness, pixelSize, radius4centering)
    #    thickness = center_CM_2darray(thickness)

    text2datfile = '# file name, Type of Fit, Curved Radius from fit [um],'
    text2datfile += ' diameter4fit [um], sigma [um], pv [um]\n'

    # %% fit radius dpc

    if False:
        dpcFiles = []
        dpcFiles.append(fname.replace('thickness', 'dpc_X'))
        dpcFiles.append(fname.replace('thickness', 'dpc_Y'))

        if len(dpcFiles) == 2:

            (dpx, pixelsize_dpc, _) = wpu.load_sdf_file(dpcFiles[0])

            (dpy, _, _) = wpu.load_sdf_file(dpcFiles[1])

            fit_radius_dpc(dpx, dpy, pixelsize_dpc,
                           radius4fit=np.min((-xx[0, 0], xx[-1, -1],
                                              -yy[0, 0], yy[-1, -1]))*.9,
                           kwave=2*np.pi/1.5498025e-10,
                           saveFigFlag=True, str4title='')

    # %% Fit

    #    nominalRadius = 23.00e-6
    wpu.print_blue('MESSAGE: Start Fit')

    if nominalRadius > 0:
        opt = [1, 2]
    else:
        opt = [1]

    for diameter4fit, i in itertools.product(diameter4fit_list, opt):

        radius4fit = _biggest_radius(thickness, pixelSize,
                                     diameter4fit/2)

        wpu.log_this('Radius of the area for fit = ' +
                     '{:.2f} um'.format(radius4fit*1e6),
                     preffname=fname2save)

        if i == 1:
            str4graphs = str4title
            (thickness_cropped, fitted,
             fitParameters) = fit_parabolic_lens_2d(thickness, pixelSize,
                                                    radius4fit=radius4fit,
                                                    mode=lensGeometry)

        elif i == 2:
            # this overwrite the previous fit, but I need that fit because it
            # is fast (least square fit) and it provides initial values for the
            # interactive fit below

            str4graphs = 'Nominal Radius Fit - ' + str4title
            p0 = [nominalRadius, fitParameters[1],
                  fitParameters[2], fitParameters[3]]
            bounds = ([p0[0]*.999999, -200.05e-6, -200.05e-6, -120.05e-6],
                      [p0[0]*1.00001, 200.05e-6, 200.05e-6, 120.05e-6])

            (thickness_cropped, fitted,
             fitParameters) = fit_nominal_lens_2d(thickness, pixelSize,
                                                  radius4fit=radius4fit,
                                                  p0=p0, bounds=bounds,
                                                  saveFigFlag=saveFigFlag,
                                                  str4title=str4graphs,
                                                  kwargs4fit={'verbose': 2,
                                                              'ftol': 1e-12,
                                                              'gtol': 1e-12})

        xmatrix, ymatrix = wpu.grid_coord(thickness_cropped, pixelSize)

        isNotNAN = np.isfinite(thickness_cropped[thickness_cropped.shape[0]//2,:])
        plot_residual_1d(xmatrix[0, isNotNAN],
                         thickness_cropped[thickness_cropped.shape[0]//2, isNotNAN],
                         fitted[thickness_cropped.shape[0]//2, isNotNAN],
                         str4title=str4graphs +
                                    '\nFit center profile Horizontal, ' +
                                    ' R = {:.4g} um'.format(fitParameters[0]*1e6),
                         saveFigFlag=True,
                         saveAsciiFlag=True)

        isNotNAN = np.isfinite(thickness_cropped[:, thickness_cropped.shape[1]//2])
        plot_residual_1d(ymatrix[isNotNAN, 0],
                         thickness_cropped[isNotNAN, thickness_cropped.shape[1]//2],
                         fitted[isNotNAN, thickness_cropped.shape[1]//2],
                         str4title=str4graphs +
                                    '\nFit center profile Vertical, ' +
                                    r' R = {:.4g} $\mu m$'.format(fitParameters[0]*1e6),
                         saveFigFlag=True,
                         saveAsciiFlag=True)

        sigma, pv = plot_residual_parabolic_lens_2d(thickness_cropped,
                                                    pixelSize,
                                                    fitted, fitParameters,
                                                    saveFigFlag=True,
                                                    savePickle=False,
                                                    str4title=str4graphs,
                                                    saveSdfData=True,
                                                    vlimErrSigma=4,
                                                    plotProfileFlag=gui_mode,
                                                    plot3dFlag=True,
                                                    makeAnimation=False)

        material = 'C'
        delta_lens = wpu.get_delta(8000, material=material, gui_mode=False)[0]
        sigma_seh, sigma_sev = slope_error_hist(thickness_cropped, fitted,
                                                pixelSize,
                                                saveFigFlag=True,
                                                delta=delta_lens,
                                                str4title=str4graphs + ' 8KeV, ' + material)

        text2datfile += fname2save
        text2datfile += ',\t Nominal'
        text2datfile += ',\t{:.4g},\t{:.4g}'.format(fitParameters[0]*1e6,
                                                    diameter4fit*1e6)
        text2datfile += ',\t{:.4g},\t{:.4g}\n'.format(sigma*1e6, pv*1e6)

    # %% write summary file

    fname_sumary = fname2save + '_2D_summary.csv'
    text_file = open(fname_sumary, 'w')
    text_file.write(text2datfile)
    text_file.close()
    wpu.print_blue('MESSAGE: Data saved at ' + fname)

    # %%
