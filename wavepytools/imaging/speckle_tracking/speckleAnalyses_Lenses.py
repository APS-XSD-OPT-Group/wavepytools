# -*- coding: utf-8 -*-  #
"""
Created on Sat Aug 13 16:00:19 2016

@author: wcgrizolli
"""

#==============================================================================
# %%
#==============================================================================
import numpy as np

from numpy.fft import fft2, ifft2, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import h5py as h5


import wavepy.utils as wpu
import wavepy.surface_from_grad as wpsg

import itertools

from scipy.optimize import curve_fit



#==============================================================================
# %% preamble
#==============================================================================


# Flags
saveFigFlag = False

# useful constants
rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias

from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc

figCount = itertools.count()  # itera
next(figCount)




# mpl_settings_4_nice_graphs()



#==============================================================================
# %% Load files
#==============================================================================

fname = wpu.select_file('**/*.h5')

f = h5.File(fname,'r')

#print(wpu.h5ListOfGroups(f))

#==============================================================================
# %% parameters
#==============================================================================

delta = 5.3265E-06
# real part refractive index Be at 8KeV from http://henke.lbl.gov/

#delta = 1.13790884E-05
# real part refractive index of Diamond at 8KeV, density = 3.5g/cm^3
# from http://henke.lbl.gov/


pixelsizeDetector = f['raw'].attrs['Pixel Size Detector [m]']

pixelsizeImg = f['displacement'].attrs['Pixel Size Processed images [m]']
distDet2sample = f['displacement'].attrs['Distance Detector to Sample [m]']
phenergy = f['displacement'].attrs['Photon Energy [eV]']

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength

print('MESSAGE: Comments from hdf5 files')
print('MESSAGE: '+ f['displacement'].attrs['Comments'])


# %%

stride = 1

pixelsizeImg = stride*pixelsizeImg

sx_raw = np.array(f['displacement/displacement_x'])[::stride,::stride]
sy_raw = np.array(f['displacement/displacement_y'])[::stride,::stride]
error_raw = np.array(f['displacement/error'])[::stride,::stride]

xVec_raw =  np.array(f['displacement/xvec'])[::stride]
yVec_raw =  np.array(f['displacement/yvec'])[::stride]

#==============================================================================
# %% Crop
#==============================================================================

idx4crop = wpu.graphical_roi_idx(np.sqrt(sx_raw**2 + sy_raw**2), verbose=True)



sx = wpu.crop_matrix_at_indexes(sx_raw, idx4crop)
sy = wpu.crop_matrix_at_indexes(sy_raw, idx4crop)
error = wpu.crop_matrix_at_indexes(error_raw, idx4crop)


xVec = wpu.realcoordvec(sx.shape[1], pixelsizeImg)
yVec = wpu.realcoordvec(sx.shape[0], pixelsizeImg)

xmatrix, ymatrix = np.meshgrid(xVec, yVec)




#==============================================================================
# %% Calculations of physical quantities
#==============================================================================


totalS = np.sqrt(sx**2 + sy**2)


# Differenctial Phase
dpx = kwave*np.arctan2(sx*pixelsizeDetector, distDet2sample)
dpy = kwave*np.arctan2(sy*pixelsizeDetector, distDet2sample)


# Differenctial Thickness
dTx = 1.0/delta*np.arctan2(sx*pixelsizeDetector, distDet2sample)
dTy = 1.0/delta*np.arctan2(sy*pixelsizeDetector, distDet2sample)




#==============================================================================
# %% integration frankotchellappa
#==============================================================================


#integration_res = frankotchellappa(dTx,dTy)
integration_res = wpsg.frankotchellappa(dTx*pixelsizeImg,dTy*pixelsizeImg)

thickness = np.real(integration_res)

thickness = thickness - np.min(thickness)

# %%

wpsg.error_integration(dTx*pixelsizeImg, dTy*pixelsizeImg, thickness,
                       pixelsizeImg, shifthalfpixel=True, plot_flag=True)

#==============================================================================
# %% Plot
#==============================================================================

def mySaveFig(figname = None):

    if figname is None:
        figname = str('output/graph_{0:02d}.png'.format(next(figCount)))


    plt.savefig(figname)
    print(figname + ' SAVED')

def mySimplePlot(array, title=''):

    plt.figure()
    plt.imshow(array, cmap='Spectral_r', interpolation='none')
    plt.title(title)
    plt.colorbar()
    if saveFigFlag: mySaveFig()
    plt.show(block=True)



def plotsidebyside(array1, array2, title1='', title2='', maintitle=''):

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(maintitle, fontsize=14)

    vmax = np.max([array1, array2])
    vmin = np.min([array1, array2])

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    im1 = ax1.imshow(array1, cmap='Spectral_r',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax1.set_title(title1, fontsize=22)
    ax1.set_adjustable('box-forced')
    fig.colorbar(im1, ax=ax1, shrink=.8, aspect=20)

    im2 = ax2.imshow(array2, cmap='Spectral_r',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax2.set_title(title2, fontsize=22)
    ax2.set_adjustable('box-forced')
    fig.colorbar(im2, ax=ax2, shrink=.8, aspect=20)


    if saveFigFlag: mySaveFig()
    plt.show(block=True)


#==============================================================================
# %% Plot dpx and dpy and fit Curvature Radius of WF
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Phase [rad]', fontsize=14)


ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)



ax1.plot(xVec*1e6, dpx[dpx.shape[1]//4,:],'-ob')
ax1.plot(xVec*1e6, dpx[dpx.shape[1]//2,:],'-or')
ax1.plot(xVec*1e6, dpx[dpx.shape[1]//4*3,:],'-og')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
ax1.set_xlabel('[um]')
ax1.set_ylabel('dpx [radians]')

lin_fitx = np.polyfit(xVec, dpx[dpx.shape[1]//2,:], 1)
lin_funcx = np.poly1d(lin_fitx)
ax1.plot(xVec*1e6, lin_funcx(xVec),'--c',lw=2)
curvrad_x = kwave/(lin_fitx[0])

ax1.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_x), fontsize=18)
ax1.set_adjustable('box-forced')


ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//4],'-ob')
ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//2],'-or')
ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//4*3],'-og')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
ax2.set_xlabel('[um]')
ax2.set_ylabel('dpy [radians]')

lin_fity = np.polyfit(yVec, dpy[:,dpy.shape[0]//2], 1)
lin_funcy = np.poly1d(lin_fity)
ax2.plot(yVec*1e6, lin_funcy(yVec),'--c',lw=2)
curvrad_y = kwave/(lin_fity[0])

ax2.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_y), fontsize=18)
ax2.set_adjustable('box-forced')



if saveFigFlag: mySaveFig()
plt.show(block=True)



# %%

plotsidebyside(sx, sy, r'Displacement $S_x$ [pixels]',
                         r'Displacement $S_y$ [pixels]')

# %%
mySimplePlot(totalS, title=r'Displacement Module $|\vec{S}|$ [pixels]')

# %%


fig = plt.figure(figsize=(14, 5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

ax1.plot(sx.flatten(),error.flatten(),'.')
ax1.set_xlabel('Sy [pixel]')
ax1.set_title('Error vs Sx', fontsize=22)
ax1.set_adjustable('box-forced')


ax2.plot(sy.flatten(),error.flatten(),'.')
ax2.set_xlabel('Sy [pixel]')
ax2.set_title('Error vs Sy', fontsize=22)
ax2.set_adjustable('box-forced')


if saveFigFlag: mySaveFig()
plt.show(block=True)


#==============================================================================
# %% Histograms to evaluate data quality
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(sx.flatten(), 51)
ax1 = plt.title(r'$S_x$ [pixels]', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(sy.flatten(), 51)
ax2 = plt.title(r'$S_y$ [pixels]', fontsize=16)


if saveFigFlag: mySaveFig()
plt.show(block=True)

##==============================================================================
## %% Total displacement
##==============================================================================
#
#plt.figure()
#plt.hist(totalS.flatten(), 51)[0]
#plt.title(r'Total displacement $|\vec{S}|$ [pixels]', fontsize=16)
#if saveFigFlag: mySaveFig()
#plt.show(block=True)


#==============================================================================
# %% Integration Real and Imgainary part
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(np.real(integration_res).flatten()*1e6, 51)
ax1 = plt.title(r'Integration Real part', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(np.imag(integration_res).flatten()*1e6, 51)
ax2 = plt.title(r'Integration Imag part', fontsize=16)

if saveFigFlag: mySaveFig()
plt.show(block=True)

# %% Crop Result and plot surface



(xVec_croped1, yVec_croped1,
 thickness_croped, _) = wpu.crop_graphic(xVec, yVec,
                                         thickness*1e6, verbose=True)

thickness_croped *= 1e-6
thickness_croped -= np.max(thickness_croped)

xmatrix_croped1, ymatrix_croped1 = wpu.realcoordmatrix_fromvec(xVec_croped1,
                                                               yVec_croped1)


# %% center fig

def center_max_2darray(array):
    '''
    crop the array in order to have the max at the center of the array
    '''
    center_i, center_j = np.unravel_index(array.argmax(), array.shape)

    if 2*center_i  > array.shape[0]:
        array = array[2*center_i-array.shape[0]:-1,:]
    else:
        array = array[0:2*center_i,:]

    if 2*center_j  > array.shape[1]:
        array = array[:, 2*center_j-array.shape[1]:-1]
    else:
        array = array[:,0:2*center_j]

    return array


# %%

thickness_croped = center_max_2darray(thickness_croped)


xVec_croped1 = wpu.realcoordvec(thickness_croped.shape[1], pixelsizeImg)
yVec_croped1 = wpu.realcoordvec(thickness_croped.shape[0], pixelsizeImg)

xmatrix_croped1, ymatrix_croped1 = np.meshgrid(xVec_croped1, yVec_croped1)



# %%

lim = 1

wpu.plot_profile(xmatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 ymatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 thickness_croped[lim:-lim,lim:-lim]*1e6,
                 title='Thickness centered [um]', xlabel='[um]', ylabel='[um]',
                 arg4main={'cmap':'Spectral_r'}) #, xo=0.0, yo=0.0)




plt.show(block=True)

# %%

#



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

stride = thickness_croped.shape[0] // 100
if stride == 0: stride = 1


surf = ax.plot_surface(xmatrix_croped1*1e6,
                       ymatrix_croped1*1e6,
                       thickness_croped*1e6,
                        rstride=stride, cstride=stride,
                        #vmin=-120, vmax=0,
                       cmap='Spectral_r', linewidth=0.1)

plt.xlabel('[um]')
plt.ylabel('[um]')

plt.title('Thickness [um]', fontsize=18, weight='bold')
plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()
if saveFigFlag: mySaveFig()
plt.show(block=True)



# =============================================================================
# %% Fits for lens
# =============================================================================

import residual_lenses as rl
import imp
imp.reload(rl)

rl.fit_radius_dpc(dpx, dpy, pixelsizeImg, 50e-6, kwave)

rl.residual_lens_1d(thickness_croped, pixelsizeImg, radius4fit=50e-6)
#
rl.residual_lens_2d(thickness_croped, pixelsizeImg, radius4fit=100e-6)


# %%

