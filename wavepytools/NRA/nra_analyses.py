#! /bin/python
# -*- coding: utf-8 -*-  #
"""
Created on Tue Oct 08

@author: wcgrizolli
"""


#import dxchange


from pywinspec import SpeFile, test_headers


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import sys

sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools/')

import wgTools as wgt


import wavepy.utils as wpu

import scipy

from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter


wpu._mpl_settings_4_nice_graphs()

#==============================================================================
# preamble
#==============================================================================


# Flags
saveFigFlag = True

# useful constants
rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias

from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc

#==============================================================================
# %% Experimental Values
#==============================================================================

pixelsize = 13.5e-6
dist2detector = 375.5e-3
phenergy = 778.00

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength


#==============================================================================
# %% Load SPE
#==============================================================================

fname = wgt.selectFile('*SPE', 3)
spe_file = SpeFile(fname)

#==============================================================================
# %% Crop ROI
#==============================================================================

#idx4crop = [635, 1993, 841, 1792]
#img = wpu.crop_matrix_at_indexes(spe_file.data[0], idx4crop)

img, idx4crop = wpu.crop_graphic_image(spe_file.data[0], verbose=False)
print(idx4crop)



img = wpu.pad_to_make_square(img, mode='edge')

# %% Plot Detector coordinates


xx, yy = wpu.realcoordmatrix(img.shape[1], pixelsize,
                             img.shape[0], pixelsize)



plt.figure(figsize=plt.figaspect(.6))
plt.contourf(xx*1e3, yy*1e3, img/np.nanmax(img), 201, cmap='plasma', vmin=0.1)
plt.xlabel(r'$x$ [mm]')
plt.ylabel(r'$y$ [mm]')
plt.title(r'Data ')
plt.colorbar()
plt.show(block=True)


#xVec = wpu.realcoordvec(img.shape[1], pixelsize)
#yVec = wpu.realcoordvec(img.shape[0], pixelsize)


# %%

#
xx, yy = wpu.realcoordmatrix(img.shape[1], 1,
                             img.shape[0], 1)
#wpu.plot_profile(xx, yy, img/np.nanmax(img))
#==============================================================================
# %% FFT
#==============================================================================


from numpy.fft import *

qx, qy = wpu.realcoordmatrix(img.shape[1], pixelsize/dist2detector*kwave,
                             img.shape[0], pixelsize/dist2detector*kwave)

fftimg = ifftshift(fft2(fftshift(img)))*pixelsize*pixelsize
abs_fftimg = np.abs(fftimg)
abs_fftimg -= np.min(abs_fftimg)
norm_abs_fftimg = abs_fftimg/np.max(abs_fftimg)

log_abs_fftimg = np.log(norm_abs_fftimg + np.finfo(float).eps)



# %%


plt.figure(figsize=plt.figaspect(.6))
plt.contourf(qx*1e-6, qy*1e-6, log_abs_fftimg, 201, cmap='plasma', vmin=-10, vmax=-.5)
plt.xlabel(r'$q_x$ [$ \mu m^{-1} $]')
plt.ylabel(r'$q_y$ [$ \mu m^{-1}$]')
plt.title(r'log of module FFT ')
plt.colorbar()
plt.show(block=True)




#==============================================================================
# %% Mask Angle
#==============================================================================

def create_mask_angle(angle, delAngle, shape, indexing='xy'):

    if indexing == 'xy':
        ii, jj = np.mgrid[shape[0]:0:-1,0:shape[1]]
        angle = - angle
    elif indexing == 'ij':
        ii, jj = np.mgrid[0:shape[0],0:shape[1]]

    ii -= shape[0] // 2
    jj -= shape[1] // 2

    #mask = 1.0*(np.logical_and(np.arctan2(ii,jj)*np.rad2deg(1) < angle + delAngle - 180.00,
    #                           np.arctan2(ii,jj)*np.rad2deg(1) > angle - delAngle - 180.00))

    #mask = 1.0*(np.logical_and(np.arctan(ii/jj)*np.rad2deg(1) < angle + delAngle,
    #                           np.arctan(ii/jj)*np.rad2deg(1) > angle - delAngle))

    mask = 1.0*(np.logical_and(np.arctan2(ii,jj)*np.rad2deg(1) < angle + delAngle,
                               np.arctan2(ii,jj)*np.rad2deg(1) > angle - delAngle) +
                np.logical_and(np.arctan2(ii,jj)*np.rad2deg(1) < 180. + angle + delAngle,
                               np.arctan2(ii,jj)*np.rad2deg(1) > 180. + angle - delAngle)+
                np.logical_and(np.arctan2(ii,jj)*np.rad2deg(1) < -180. + angle + delAngle,
                               np.arctan2(ii,jj)*np.rad2deg(1) > -180. + angle - delAngle))

    #mask = 1.0*(np.logical_and(jj/ii < np.tan(angle + delAngle),
    #                           jj/ii > np.tan(angle - delAngle)) +
    #            np.logical_and(jj/ii < np.tan(180. + angle + delAngle),
    #                           jj/ii > np.tan(180. + angle - delAngle)))

    mask[mask>.5] = 1.000
    mask[mask<.5] = np.nan

    return mask


plotThis = log_abs_fftimg
plotThis = plotThis[::-1,:]

# Select angle


#joio = wpu.graphical_select_point_idx(plotThis, verbose=True)
#jo = int(joio[0])
#io = int(joio[1])
#angle = np.arctan2(abs_fftimg.shape[0]//2 - io, jo - abs_fftimg.shape[1]//2)*np.rad2deg(1)

angle = -21.4061120849

print('angle = ' + str(angle))



# %%
mask_angle = create_mask_angle(angle, 1, abs_fftimg.shape, indexing = 'xy')


print('oi 1346')


# %% peaks

def create_mask_peaks2DOLD(array2D, threshold=None, order=3):


    import scipy.signal

    if threshold is not None:
        mask_threshold = wpu.nan_mask_threshold(array2D, threshold=.001)
    else:
        mask_threshold = array2D*0.0 +1.0



    idx_x_axis_0, idx_y_axis_0 = scipy.signal.argrelmax(array2D*mask_threshold,
                                                        axis=0, order = order)
    idx_x_axis_1, idx_y_axis_1 = scipy.signal.argrelmax(array2D*mask_threshold,
                                                        axis=1, order = order)

    peaks_axis0 = np.zeros(np.shape(array2D))
    peaks_axis0[idx_x_axis_0[:], idx_y_axis_0[:]] = 1.0
    peaks_axis1 = np.zeros(np.shape(array2D))
    peaks_axis1[idx_x_axis_1[:], idx_y_axis_1[:]] = 1.0



    return peaks_axis0, peaks_axis1

# %%

def create_mask_peaks2D(array2D, order=1, mode='clip', srn_threshold=0.0):


    array2D = np.pad(array2D[order:-order,order:-order], order, 'edge')
    # make our life easier by making the edges peak free

    idx_axis_0 = scipy.signal.argrelmax(array2D, axis=0, order=order, mode=mode)
    idx_axis_1 = scipy.signal.argrelmax(array2D, axis=1, order=order, mode=mode)

    peaks_axis0 = np.zeros(np.shape(array2D))
    peaks_axis0[idx_axis_0] = 1.0
    peaks_axis1 = np.zeros(np.shape(array2D))
    peaks_axis1[idx_axis_1] = 1.0


    snr0 = np.zeros(np.shape(array2D))
    snr1 = np.zeros(np.shape(array2D))

    snr0[idx_axis_0] = np.abs(array2D[idx_axis_0[0], idx_axis_0[1]] / \
                              np.mean((array2D[idx_axis_0[0] - order, idx_axis_0[1]],
                              array2D[idx_axis_0[0] + order, idx_axis_0[1]])))

    snr1[idx_axis_1] = np.abs(array2D[idx_axis_1[0], idx_axis_1[1]] / \
                              np.mean((array2D[idx_axis_1[0], idx_axis_1[1]],
                              array2D[idx_axis_1[0], idx_axis_1[1]+ order])))

    srn = (snr0 + snr1)/2
    mask_snr = np.where(srn > srn_threshold, 1, 0)


    return np.where(peaks_axis0*peaks_axis1*mask_snr >= 0.5), srn*peaks_axis0*peaks_axis1


# %%

for srn_threshold in [1, 1.5, 2, 3, 5]:

    fig = plt.figure(figsize=plt.figaspect(.6))




    plotThis = log_abs_fftimg

    plt.contourf(qx, qy, plotThis, 101, cmap='plasma', vmin=-10, vmax=5)


    [idx_angle_i, idx_angle_j], srn = create_mask_peaks2D(gaussian_filter(norm_abs_fftimg, 2),
                                                     order=4, srn_threshold=srn_threshold)

    #[idx_angle_i, idx_angle_j], srn =  create_mask_peaks2D(norm_abs_fftimg,
    #                                                       order=5, srn_threshold=2.0)

    plt.plot(qx[idx_angle_i,idx_angle_j],
             qy[idx_angle_i,idx_angle_j], 'bx', ms=10, mew=2)



    plt.show(block=False)

# %%


fig = plt.figure(figsize=plt.figaspect(.6))


plotThis = maximum_filter(srn, 3)

plotThis[plotThis > 500] = NAN

plt.contourf(qx, qy, plotThis, 101, cmap='plasma')

plt.colorbar()
plt.show(block=True)



# %%

plt.figure()
plotThis = log_abs_fftimg
plt.contourf(qx*1e-3, qy*1e-3, plotThis, 101, cmap='plasma', vmin=-10, vmax=5)

idx_angle_i, idx_angle_j = np.where(peaks_mask0*peaks_mask1>.5)
plt.plot(qx[idx_angle_i,idx_angle_j]*1e-3,
         qy[idx_angle_i,idx_angle_j]*1e-3, 'bx', ms=10, mew=2)

plt.title('FFT Image, Log Scale')
plt.colorbar()
plt.show(block=True)


print('oi')

# %%

fig = plt.figure(figsize=plt.figaspect(.6))

plt.contourf(qx*1e-6, qy*1e-6, log_abs_fftimg, 201, cmap='plasma', vmin=-10, vmax=-.5)
plt.xlabel(r'$q_x$ [$ \mu m^{-1} $]')
plt.ylabel(r'$q_y$ [$ \mu m^{-1}$]')
plt.title(r'log of module FFT ')
plt.colorbar()


idx_angle_i, idx_angle_j = np.where(peaks_mask0*peaks_mask1*mask_angle>.5)
plt.plot(qx[idx_angle_i,idx_angle_j]*1e-6,
         qy[idx_angle_i,idx_angle_j]*1e-6, 'bo', ms=10, mew=2, mfc="None")



plt.show(block=True)


# %%


import skimage.filters
import scipy.ndimage


scipy.ndimage.uniform_filter


# %%

vec_q = np.sqrt(qx[idx_angle_i,idx_angle_j]**2 + \
                qy[idx_angle_i,idx_angle_j]**2)*np.sign(qy[idx_angle_i,idx_angle_j])

#intensity = abs_fftimg[idx_angle_i,idx_angle_j]
#intensity = gaussian_filter(abs_fftimg, 2.5)[idx_angle_i,idx_angle_j]


intensity = scipy.ndimage.uniform_filter(abs_fftimg, 5)[idx_angle_i,idx_angle_j]



# %%

plt.figure()

plt.plot(vec_q, intensity, '-x', ms=10, mew=2)
plt.plot(-vec_q, intensity*1.1, '-x', ms=10, mew=2)

plt.xlabel(r'q [$ m^{-1}$]')
plt.title('FFT Image, Log Scale')
plt.show(block=True)


# %%

rho_x, rho_y = wpu.reciprocalcoordmatrix(qx.shape[1], qy[1, 0] - qy[0,0],
                                         qy.shape[0], qx[0, 1] - qx[0,0])



rho_x *= 2*np.pi
rho_y *= 2*np.pi


vec_rho = np.sqrt(rho_x[idx_angle_i,idx_angle_j]**2 + \
                  rho_y[idx_angle_i,idx_angle_j]**2)*np.sign(rho_y[idx_angle_i,idx_angle_j])




# %%
plt.figure()


intensity = scipy.ndimage.uniform_filter(norm_abs_fftimg, 3)[idx_angle_i,idx_angle_j]
plt.plot(vec_rho*1e6, intensity, '-xg', ms=10, mew=2)

intensity = scipy.ndimage.uniform_filter(norm_abs_fftimg, 0)[idx_angle_i,idx_angle_j]
plt.plot(vec_rho*1e6, intensity, '-xb', ms=10, mew=2)



print(vec_rho*1e6)

for i in range(intensity.size):

    print('{:.3f}, \t {:2.4g}'.format(vec_rho[i]*1e6, intensity[i]*1e2))


intensity = scipy.ndimage.uniform_filter(norm_abs_fftimg, 5)[idx_angle_i,idx_angle_j]
plt.plot(vec_rho*1e6, intensity, '-xr', ms=5, mew=2)


#plt.plot(-vec_rho*1e6/1.5, intensity*1.1, '-x', ms=10, mew=2)


plt.xlabel(r'$\rho$  [$\mu m$]')
plt.title('FFT Image, Log Scale')
plt.show(block=True)



# %%

exit()

# %%
c_j_over_c_0 = np.array([4.465, 3.976, 3.13, 3.024, 3.113, 2.308, 1.781])

rho = np.array([1.5,   3. ,   4.5,   6. ,   7.5,   9. ,  10.5])


S_j = np.array([0.992, 0.968, 0.96, 0.81, 0.817, 0.576, 0.791])

sigma = 4

intensity_temp = c_j_over_c_0/S_j



# %%
plt.figure()


#
#plt.plot(rho, c_j_over_c_0, '-or', ms=10, mew=2)

#plt.plot(rho, S_j, '-ob', ms=10, mew=2)
plt.plot(rho, intensity_temp, '-kx', ms=10, mew=2)


plt.xlabel(r'$\rho$  [$\mu m$]')
plt.show(block=False)



# %%



