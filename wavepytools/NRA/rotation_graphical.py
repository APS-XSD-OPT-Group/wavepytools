# -*- coding: utf-8 -*-  #
"""
Created on Sat Oct  8 16:40:46 2016

@author: grizolli
"""


import numpy as np
from numpy.fft import *
import wavepy.utils as wpu
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy
import scipy.signal
import scipy.ndimage
import skimage.transform
from scipy.ndimage import gaussian_filter

import skimage.filters



from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc

#==============================================================================
# %% Experimental Values
#==============================================================================

pixelsize = 13.5e-6
dist2detector = 390.5e-3
phenergy = 778.00

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength


#==============================================================================
# %% Load SPE
#==============================================================================


from pywinspec import SpeFile, test_headers

fname = wpu.select_file('**/*SPE', 3)
spe_file = SpeFile(fname)


#==============================================================================
# %% Crop ROI
#==============================================================================

img, _ = wpu.crop_graphic_image(spe_file.data[0])




#==============================================================================
# %% Padding for square image
#==============================================================================

img = wpu.pad_to_make_square(img, mode='edge')


# %% FFT


fftimg = ifftshift(fft2(fftshift(img)))
abs_fftimg = np.abs(fftimg)
abs_fftimg -= np.min(abs_fftimg)
norm_abs_fftimg = abs_fftimg/np.nanmax(abs_fftimg)

img2 = np.log(norm_abs_fftimg + 1e-10)





# %% Rotation in the Reciprocal space

img_rotated, angle = wpu.rotate_img_graphical(img2, mode='edge')

norm_abs_fftimg = skimage.transform.rotate(norm_abs_fftimg, -angle,
                                           mode='edge')


# %% x and y axes

qx, qy = wpu.realcoordmatrix(img_rotated.shape[1], pixelsize/dist2detector*kwave,
                             img_rotated.shape[0], pixelsize/dist2detector*kwave)

rho_x, rho_y = wpu.reciprocalcoordmatrix(qx.shape[1], qy[1, 0] - qy[0,0],
                                         qy.shape[0], qx[0, 1] - qx[0,0])



rho_x *= 2*np.pi
rho_y *= 2*np.pi


# %%


#plotThis = norm_abs_fftimg # *wpu.nan_mask_threshold(norm_abs_fftimg,1e-2*1j)
#plotThis = scipy.ndimage.uniform_filter(norm_abs_fftimg, size=(1,1))


plotThis = scipy.ndimage.uniform_filter(norm_abs_fftimg, size=(1,1))
plotThis[plotThis >= .01] = 0.0


wpu.plot_profile(rho_x, rho_y, plotThis,
                 arg4main={'vmax':1e-2})

plt.show(block=True)


# %%

def create_mask_peaks2D(array2D, threshold=None, order=3):



    if threshold is not None:
        mask_threshold = wpu.nan_mask_threshold(array2D, threshold=threshold)
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



# %% Peaks


peaks_mask0, peaks_mask1 = create_mask_peaks2D(img_rotated[551:553,:],
                                               order=2, threshold=.0001)


idx_angle_i, idx_angle_j = np.where(peaks_mask0*peaks_mask1>.5)

idx_theoretical_peaks = []
idx_theoretical_peaks2 = []
idx_theoretical_peaks3 = []

idy_theoretical_peaks = []
idy_theoretical_peaks2 = []
idy_theoretical_peaks3 = []

for val in [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12., 13.5, 15., 19.5, 21.0, 22.5, 25.5, 34.5]:

    idx_theoretical_peaks.append(wpu.find_nearest_value_index(rho_x[0,:], val*1e-6)[0])
    idx_theoretical_peaks2.append(wpu.find_nearest_value_index(rho_x[0,:], val/2*1e-6)[0])
    idx_theoretical_peaks3.append(wpu.find_nearest_value_index(rho_x[0,:], val/3*1e-6)[0])

    idy_theoretical_peaks.append(wpu.find_nearest_value_index(rho_y[:,0], val*1e-6)[0])
    idy_theoretical_peaks2.append(wpu.find_nearest_value_index(rho_y[:,0], val/2*1e-6)[0])
    idy_theoretical_peaks3.append(wpu.find_nearest_value_index(rho_y[:,0], val/3*1e-6)[0])



idx_theoretical_peaks = np.array(idx_theoretical_peaks)
idx_theoretical_peaks2 = np.array(idx_theoretical_peaks2)
idx_theoretical_peaks3 = np.array(idx_theoretical_peaks3)



idy_theoretical_peaks = np.array(idy_theoretical_peaks)
idy_theoretical_peaks2 = np.array(idy_theoretical_peaks2)
idy_theoretical_peaks3 = np.array(idy_theoretical_peaks3)

#==============================================================================
# %% Filter
#==============================================================================

plt.figure()
plotThis = norm_abs_fftimg[img_rotated.shape[0] // 2,:]

plt.plot(rho_x[0,:]*1e6, plotThis[:], '-xk', label='data')

plt.plot(rho_x[0,idx_theoretical_peaks]*1e6,
         plotThis[idx_theoretical_peaks], 'dr', ms=15, label='1st order peaks')


plt.plot(rho_x[0,idx_theoretical_peaks2]*1e6,
         plotThis[idx_theoretical_peaks2], 'sb', ms=10, label='1st order peaks')

plt.plot(rho_x[0,idx_theoretical_peaks3]*1e6,
         plotThis[idx_theoretical_peaks3], 'og', label='3rd order peaks')

plt.legend()
plt.show(block=True)


# %%
plt.figure()
plotThis = norm_abs_fftimg[:, img_rotated.shape[1] // 2]

plt.plot(rho_y[:, 0]*1e6, plotThis[:], '-xk', label='data')

plt.plot(rho_y[idy_theoretical_peaks, 0]*1e6,
         plotThis[idy_theoretical_peaks], 'dr', ms=15, label='1st order peaks')


plt.plot(rho_y[idy_theoretical_peaks2, 0]*1e6,
         plotThis[idy_theoretical_peaks2], 'sb', ms=10, label='1st order peaks')

plt.plot(rho_y[idy_theoretical_peaks3, 0]*1e6,
         plotThis[idy_theoretical_peaks3], 'og', label='3rd order peaks')

#plt.plot(skimage.filters.gaussian(plotThis, 5//2)[img_rotated.shape[0] // 2,:],
#         '-ro', label='gauss filter')
#
#plt.plot(scipy.ndimage.uniform_filter(plotThis, size=(20,1))[img_rotated.shape[0] // 2,:],
#         '-bo', label='uniform filter')
#
plt.legend()
plt.show(block=True)

# %%


plt.figure()


for sigma in [0, 2, 4, 6]:
    #for sigma in [3]:

    plotThis = skimage.filters.gaussian(norm_abs_fftimg,sigma)[img_rotated.shape[0] // 2, :]
    plotThis /= np.max(plotThis)

    idx_x = scipy.signal.argrelmax(plotThis, order = 10)



    plt.plot(rho_y[:,0]*1e6,
             plotThis, '-xk', lw=2)

    plt.plot(rho_x[0,idx_x][0]*1e6,
             plotThis[idx_x], '-d', ms=10)

    plt.xlim((1,15))
    plt.ylim((-.005,.1))
    plt.xlabel(r'Pinholes Separation [$\mu$ m]')
    plt.ylabel(r'FFT amplitude [a. u.]')
    plt.grid()
    plt.title('Gaussian Filters, sigma = {:d} pixels'.format(sigma))
    plt.savefig(str('sigma_{:d}.png'.format(sigma)))
    plt.show(block=True)
#plt.legend(['no filter', 'no filter', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5'])
#plt.legend(['no filter', 'no filter', '2', '2', '4', '4', '6', '6'])



