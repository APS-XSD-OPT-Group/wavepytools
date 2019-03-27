#! /bin/python
# -*- coding: utf-8 -*-  #
"""
Created 20161014

@author: wcgrizolli
"""


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sys
sys.path.append('/home/grizolli/workspace/MAXIVpythonWorkspace/wgTools')

from myFourierLib import *


import skimage
import skimage.filters


from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc


import wavepy.utils as wpu

def gaussianBeamEquation(fwhm, wavelength, x, y, z):
    '''
    Create gaussian beam acconrding to equation 3.1-7 from Saleh
    '''

    # equation 3.1-7 from Saleh
    Wo = fwhm*0.84932180028801907  # Wo = 2 sigma
    zo = np.pi*Wo**2/wavelength
    Wz = Wo*np.sqrt(1.0 + (z/zo)**2)
#    Rz = z*(1.0 + (zo/z)**2)
    inverseRz = z/(z**2 + zo**2)  # inverse of Rz, to avoid division by zero
    zeta = np.arctan(z/zo)
    k = 2.0*np.pi/wavelength

    rho2 = x**2 + y**2  # rho**2

    return Wo/Wz*np.exp(-rho2/Wz**2)*np.exp(-1j*k*z-1j*k*rho2/2*inverseRz+1j*zeta)

# %%
def gaussianBeam2D(x, y, z, fwhm_x, fwhm_y, phEnergy,
                   xo=0.0, yo=0.0,
                   focus_x=0.0, focus_y=0.0):


    '''
    Stigmatic Elliptical Gaussian beam with different vert and hor sizes

    Parameters
    ----------
    fwhm_x, fwhm_y: float
        FWHM of the INTENSITY (not the EM field) of gaussian beam in horizontal and vertical
        directions. Note that the beam waist is equal to 2*sigma (2 times
        standart deviation), and that FWHM = 2.25*sigma #TODO: math format

    phEnergy: float
        Photon Energy in electronvolts

    x, y: ndarray
        `x` and `y` coordinate matrices

    z: float
        Propagation distance. Note that, when focus_x=focus_y=0.0, z=0.0 is
        the postion of the beam waist


    focus_x, focus_y: float
        beam waist position in ``x`` and ``y`` directions.

    Note
    ----
    Follows notation  used for the gaussian beam in equation 3.1-7 of Saleh,
    excepted that that equation is for an stigmatic and radial symetric beam.
    See this article for the full equation:
    http://dx.doi.org/10.1364/AO.52.006030

     '''


    # equation 3.1-7 from Saleh

    wavelength = hc/phEnergy

    Wxo = fwhm_x/np.sqrt(2*np.log(2))
    Wyo = fwhm_y/np.sqrt(2*np.log(2))

    zxo = np.pi*Wxo**2/wavelength
    zyo = np.pi*Wyo**2/wavelength

    Wxz = Wxo*np.sqrt(1.0 + ((z - focus_x)/zxo)**2)
    Wyz = Wyo*np.sqrt(1.0 + ((z - focus_y)/zyo)**2)


    inverseRxz = (z - focus_x)/((z - focus_x)**2 + zxo**2)
    inverseRyz = (z - focus_y)/((z - focus_y)**2 + zyo**2)

    inverse_qx = inverseRxz - 1j*wavelength/np.pi/Wxz**2
    inverse_qy = inverseRyz - 1j*wavelength/np.pi/Wyz**2

    zeta = ( np.arctan((z - focus_x)/zxo) + \
             np.arctan((z - focus_x)/zyo) )/2

    k = 2.0*np.pi/wavelength

    return np.sqrt(inverse_qx*inverse_qy)* \
           np.exp(-1j*k*(x-xo)**2*inverse_qx)* \
           np.exp(-1j*k*(y-yo)**2*inverse_qy)* \
           np.exp(-1j*k*z)*np.exp(1j*zeta)




# %%

pixelsize = .1e-6
dist2detector = 450e-3
phenergy = 778.00

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength


# %% Make the mask
positions_x = np.array([-22.5, -7.5, -1.5, 0, 3, 12, 0, 0, 0, 0, 0])*1e-6
positions_y = np.array([0, 0, 0, 0, 0, 0, -22.5, -7.5, -1.5, 3, 12])*1e-6



xx, yy = wpu.realcoordmatrix(2001, pixelsize, 2001, pixelsize)

# abs mask
mask_NRA = xx*0.0

for i in range(positions_x.size):

    xo, yo = positions_x[i], positions_y[i]
    mask_NRA[(np.sqrt((xx - xo)**2+(yy - yo)**2) < .50000000e-6)] += 1.0


# phase mask
#mask_NRA = xx*0.0*1j + 1.0
#
#for i in range(positions_x.size):
#
#    xo, yo = positions_x[i], positions_y[i]
#    mask_NRA[(np.sqrt((xx - xo)**2+(yy - yo)**2) < .4e-6)] *= np.exp(1j*1.0000*np.pi)


# %%

#plt.figure()
#
#plt.contourf(xx*1e6, yy*1e6, mask_NRA, 101)
#
#plt.show()


# %%

fwhm_x = 100e-6
fwhm_y = 20e-6


beam_profile = gaussianBeam2D(xx, yy, 0.0, fwhm_x, fwhm_y, phenergy,
                              xo=0.0, yo=0.0,
                              focus_x=0.0, focus_y=0.0)


# %%


plt.figure()

plt.imshow(np.abs(beam_profile))

plt.show()

# %%

#beam_profile = np.exp(-(xx/fwhm_x)**2-(yy/fwhm_y)**2)

#beam_profile = 1./(np.pi*fwhm_x*(1.+(xx-0.0)**2/fwhm_x**2))/(np.pi*fwhm_y*(1.+(yy-0.0)**2/fwhm_y**2))

beam_profile /= np.max(np.abs(beam_profile))

sample = mask_NRA *beam_profile


# %%

def beam_stopper(xx, yy, radius=1):
    array = xx*0.0 + 1.0

    array[(np.sqrt(xx**2+yy**2) < radius)] = 0.0

    return array


# %%




L2 = 5e-3
u2 = prop2step(sample,
               mask_NRA.shape[1]*pixelsize,
               L2,
               wavelength,450e-3)


at_detector2 = np.abs(u2)



xx2, yy2 = wpu.realcoordmatrix(at_detector2.shape[1], L2/at_detector2.shape[1],
                               at_detector2.shape[0], L2/at_detector2.shape[0])


# %%
plt.figure()
plt.imshow(np.abs(at_detector2*beam_stopper(xx2, yy2, .1e-3)))

plt.show(block=True)


## %%
#
#wpu.plot_profile(xx2, yy2, at_detector2)
#
#
#plt.show(block=True)

# %%


fftimg = ifftshift(fft2(fftshift(at_detector2*beam_stopper(xx2, yy2, .1e-3))))


# %%
plt.figure()
plt.imshow(np.abs(fftimg)*beam_stopper(xx2, yy2, .02e-3))

plt.show(block=True)

# %%

qx2, qy2 = wpu.realcoordmatrix(at_detector2.shape[1],
                               L2/at_detector2.shape[1]/dist2detector*kwave,
                               at_detector2.shape[0],
                               L2/at_detector2.shape[0]/dist2detector*kwave)

rho_x, rho_y = wpu.reciprocalcoordmatrix(qx2.shape[1],
                                         qy2[1, 0] - qy2[0,0],
                                         qy2.shape[0],
                                         qx2[0, 1] - qx2[0,0])


rho_x *= 2*np.pi
rho_y *= 2*np.pi

#
#qx2 = 2*kwave*np.sin(1./2*np.arctan2(xx2,dist2detector)) + 1e-20
#qy2 = 2*kwave*np.sin(1./2*np.arctan2(yy2,dist2detector)) + 1e-20
#
#rho_x, rho_y = 1/qx2, 1/qy2



# %%

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







# %%

plt.figure()

line = fftimg.shape[0] //2 +1

norm = 1/np.max(np.abs(fftimg))



plotThis = skimage.filters.gaussian(np.abs(fftimg)*norm,2)

plt.plot(rho_x[line,:]*1e6,
             plotThis[line,:], '-+k', label='data hor')


plt.plot(rho_x[line,idx_theoretical_peaks]*1e6,
         plotThis[line,idx_theoretical_peaks],
         'dr', ms=10, label='1st order peaks')


plt.plot(rho_x[line,idx_theoretical_peaks2]*1e6,
         plotThis[line,idx_theoretical_peaks2],
        'sb', label='2nd order peaks')

plt.plot(rho_x[line,idx_theoretical_peaks3]*1e6,
         plotThis[line,idx_theoretical_peaks3],
        'og', label='3rd order peaks')

plt.legend()
plt.show()


# %%

plt.figure()

line = fftimg.shape[1] //2 +1





plt.plot(rho_y[:, line]*1e6,
             plotThis[:, line], '-+r', label='data vert')



plt.plot(rho_y[idy_theoretical_peaks, 0]*1e6,
         plotThis[idy_theoretical_peaks,line], 'dk', ms=15, label='1st order peaks')


plt.plot(rho_y[idy_theoretical_peaks2, 0]*1e6,
         plotThis[idy_theoretical_peaks2, line], 'sb', ms=10, label='1st order peaks')

plt.plot(rho_y[idy_theoretical_peaks3, 0]*1e6,
         plotThis[idy_theoretical_peaks3, line], 'og', label='3rd order peaks')





plt.legend()
plt.show()



# %%







