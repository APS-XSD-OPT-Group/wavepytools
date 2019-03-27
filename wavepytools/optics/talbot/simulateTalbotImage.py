#! /bin/env python
# -*- coding: utf-8 -*-  #
"""
Created on Jan 30, 2018

@author: wcgrizolli
"""


# %% ATTENTION: This code is a mess. I promisse to clean it before Ragnarok

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

import sys
sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools')
from myOpticsLib import *
from myFourierLib import *

import wavepy.utils as wpu
from unwrap import unwrap

from libtiff import TIFF
import scipy.misc
from scipy.ndimage.filters import gaussian_filter

print(sys.argv)

# %% define some functions


def tFuncLens(X, Y, wavelength, fx=1e23, fy=1e23, D=None, valOutLens=None):

    lensFunc = np.exp(-1j*2*np.pi/wavelength/2/fx*(X**2+Y**2))

    if D is not None:
        if valOutLens is None:
            lensFunc[X**2 + Y**2 >= D**2/4] = np.exp(-1j*2*np.pi/wavelength/2/fx*(D**2/4))
        else:
            lensFunc[X**2 + Y**2 > D**2/4] = (1+0j)*valOutLens

    return lensFunc


def save_tiff(fname, array, maxValue=65535):

    tiff = TIFF.open(fname, mode='w')
    norm = array.max()
    if int(norm) == 0:
        norm = 1

    tiff.write_image(np.asarray(array/norm*maxValue, dtype=np.uint16))
    tiff.close()

    print(fname + ' saved')



def noise(array, amp=.01):

    return (np.random.rand(array.shape[0], array.shape[1])*array.ptp()*amp)

# %% initial parameters
plotFlag = False

phEnergy = 8e3
wavelength = wpu.hc/phEnergy
grPeriod = 4.8e-6
np_gr = 48 # number of points within the grating period

#npoints = int(Lx/grPeriod*np_gr)

npoints = 2**12
Lx = Ly = npoints*grPeriod/np_gr
#Lx = Ly = 409.6e-6


print('Lx = {:.3f} um'.format(Lx*1e6))
print('npoints period = {:d}'.format(np_gr))
print('npoints total = {:d}'.format(npoints))
print('grating lateral size = {:.2f} um'.format(grPeriod*np_gr*1e6))

zt = (grPeriod/2)**2/wavelength
Y, X = np.mgrid[-Lx/2:Lx/2:1j*npoints, -Lx/2:Lx/2:1j*npoints]

# %% Beam 1 Plane wave

#sigma_x = sigma_y = 0.5e-3
#emSource = ((1j)*X*0.0 +1.0)*np.exp(-X**2/sigma_x**2 - Y**2/sigma_y**2)
#sourceStr = 'planeWave'

# %% Beam 2 Spherical Wave

sourcedx = 30.00
#sourcedy = sourcedx
#sourceStr = str('sphWave{:.2f}'.format(sourcedx))

sourcedy = 35.00 # negative: convergent beam
sourceStr = str('AstSphWave{:.2f}x_{:.2f}y'.format(sourcedx, sourcedy))

sigma_x = sigma_y = 0.5e-3
phase = 2*np.pi/wavelength*(X**2/sourcedx/2 + Y**2/sourcedy/2)

emSource = (np.cos(phase) + 1j*np.sin(phase))*np.exp(-X**2/sigma_x**2 - Y**2/sigma_y**2)


# %% Beam 3 Gaussian Beam

#div_x = 50e-6
#div_y = 1e-6
#
#fwhm_x = wavelength/4/np.pi/div_x
#fwhm_y = wavelength/4/np.pi/div_y
#emSource = gaussianBeamAst(X, Y, fwhm_x, fwhm_y, z=5.0, zxo=-10, zyo=0, wavelength=wavelength)

# %% Source intensenty and wf


#wf = unwrap(np.angle(emSource))

# %% Plots

if plotFlag:

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X*1e3, Y*1e3, np.abs(emSource),
                           rstride=npoints//100, cstride=npoints//100,
                           cmap='jet',
                           linewidth=0, antialiased=False)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X*1e3, Y*1e3, wf/2/np.pi,
                           rstride=npoints//100, cstride=npoints//100,
                           cmap='jet',
                           vmin=np.nanmin(wf/2/np.pi), vmax=np.nanmax(wf/2/np.pi),
                           linewidth=0, antialiased=False)
    plt.show()

# %% Get source Curvature radius from data
#
dx = X[0,1] - X[0,0]
dy = Y[1,0] - Y[0,0]


#dx = dy = 1e-6

if 'plane' not in sourceStr:


    wf_wraped = np.angle(emSource)

    d2z_dx2 = np.diff(wf_wraped, 2, 1)/dx**2
    d2z_dy2 = np.diff(wf_wraped, 2, 0)/dy**2

    Rx = 2*np.pi/wavelength/d2z_dx2
    Ry = 2*np.pi/wavelength/d2z_dy2


    print('Rx: {:.4f}, sdv: {:.4g}'.format(np.nanmean(Rx), np.nanstd(Rx)))
    print('Ry: {:.4f}, sdv: {:.4g}'.format(np.nanmean(Ry), np.nanstd(Ry)))

    wf_wraped = None

# %% clear memory
phase = None
Rx = Ry = None
d2z_dx2 = d2z_dy2 = None


# %% Create Grating

#checkerboard = wpu.dummy_images('Checked', shape=(npoints//2, npoints//2),
#                                nLinesH=Lx/2/grPeriod, nLinesV=Lx/2/grPeriod)

checkerboard = wpu.dummy_images('Checked', shape=(npoints, npoints),
                                nLinesH=Lx/grPeriod, nLinesV=Lx/grPeriod)

gr = np.exp(-1j*np.pi*checkerboard) # Pi phase grating
#gr = np.exp(-1j*np.pi/3*checkerboard) # Pi/2 phase grating
#gr[gr<0.1] = 0.1 + 0j

marginx = (npoints - gr.shape[0])//2
marginy = (npoints - gr.shape[1])//2

gr = np.pad(gr, ((marginx, marginx),
                 (marginy, marginy)), mode='constant', constant_values=(1,))
gr = np.pad(gr, ((0, npoints - gr.shape[0]),
                 (0, npoints - gr.shape[1])), mode='constant', constant_values=(1,))


emgr = emSource*gr

intensitygr = np.abs(gr)
wfgr = unwrap(np.angle(gr))



# %% Propagation 1

nTalbot = int(sys.argv[1])
d_sample_gr = float(sys.argv[2])


zz = np.round(zt*(nTalbot-1/2), 3)  # nth talbot distance
#zz = 1

fresnelNumber(grPeriod, zz, wavelength, verbose=True)

#start_time = time.time()
#u2_xy = propTForIR(emgr, Lx, Ly, wavelength, zz)
#wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))

#
#u2_xy = propIR_RayleighSommerfeld(emgr,Lx,Ly,wavelength,zz)
#titleStr = str(r'propIR_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
#               % (zz*1e3))







start_time = time.time()
em_b4_gr = propTF_RayleighSommerfeld(emSource, Lx, Ly, wavelength, d_sample_gr)
wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))



start_time = time.time()
u2_xy = propTF_RayleighSommerfeld(em_b4_gr*gr,Lx,Ly,wavelength,zz)
wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))


print('Propagation Done!')

intensityDet = np.abs(u2_xy)
wfdet = unwrap(np.angle(u2_xy))
u2_xy = None
# %% Plots

if plotFlag:
    #wpu.plot_profile(X[800:1200, 800:1200],
    #                 Y[800:1200, 800:1200],
    #                 intensityDet[800:1200, 800:1200])
    plt.figure()
    plt.imshow(intensityDet, interpolation='none')
    plt.show()

    plt.figure()
    plt.plot(X[0, :]*1e6, intensityDet[1000,:], '-o')
    plt.show()

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X*1e3, Y*1e3, intensityDet,
    #                       rstride=npoints//100, cstride=npoints//100,
    #                       cmap='jet',
    #                       linewidth=0, antialiased=False)
    #plt.show()


# %% Create lens
delta = wpu.get_delta(phEnergy, 1, gui_mode=False)[0]

lensRadius = 50e-6  # of one face
nlenses = 1
fx = lensRadius/2/nlenses/delta
tLens = tFuncLens(X, Y, wavelength, fx=fx, D=340e-6)
wfLens = unwrap(np.angle(emSource*tLens))
d2z_dx2 = np.diff(wfLens, 2, 1)/dx**2
d2z_dy2 = np.diff(wfLens, 2, 0)/dy**2

Rx = 2*np.pi/wavelength/d2z_dx2
Ry = 2*np.pi/wavelength/d2z_dy2


print('Rx: {:.4f}, sdv: {:.4g}'.format(np.nanmean(Rx), np.nanstd(Rx)))
print('Ry: {:.4f}, sdv: {:.4g}'.format(np.nanmean(Ry), np.nanstd(Ry)))


if plotFlag:

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X*1e3, Y*1e3, wfLens,
                           rstride=npoints//100, cstride=npoints//100,
                           cmap='jet',
                           linewidth=0, antialiased=False)
    plt.show()

# %% Porpagate lens emf

# propagation lens to grating, d_gr_lens ~= 100mm
start_time = time.time()
emLen = propTF_RayleighSommerfeld(emSource*tLens, Lx, Ly, wavelength, d_sample_gr)
wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))


start_time = time.time()
u2_xy = propTF_RayleighSommerfeld(emLen*gr, Lx, Ly, wavelength, zz)
wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))

print('Propagation Done!')
intensitylensdet = np.abs(u2_xy)
wflensdet = unwrap(np.angle(u2_xy))

if plotFlag:

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X*1e3, Y*1e3, wfdet,
                           rstride=npoints//100, cstride=npoints//100,
                           cmap='jet',
                           linewidth=0, antialiased=False)
    plt.show()


    from fastplot_with_pyqtgraph import plot_surf_fast

    plot_surf_fast(wflensdet[::1,::1]-np.min(wflensdet))

    #plot_surf_fast(intensitylensdet[::10,::10]-np.min(intensitylensdet))

# %% Save images


#fname = str(sourceStr + '_{:.0f}eV_d_{:.1f}mm_pixel_{:.3f}um_grPeriod_{:.1f}um'.format(phEnergy, zz*1e3, dx*1e6, grPeriod*1e6))
#fname = fname.replace('.', 'p')
#fname += '.tif'
#
#save_tiff('1_image_' + fname, intensitylensdet)
#save_tiff('2_ref_' + fname, intensityDet)
#save_tiff('3_dark_'+ fname, intensityDet*0.0)
#save_tiff('4_gratingPhase_'+ fname, np.angle(gr))
#
#print('Done!')

# %% Save images - rescale

fname = str(sourceStr + '_{:.0f}eV_d_{:.1f}mm_pixel_{:.3f}um_grPeriod_{:.1f}um'.format(phEnergy, zz*1e3, dx*1e6*5, grPeriod*1e6))
fname = fname.replace('.', 'p')
fname += '.tif'
#
#from skimage.transform import rescale, downscale_local_mean, resize
#
#
#save_tiff('rescale1_1_image_' + fname, rescale(intensitylensdet, 0.2, mode='constant'))
#save_tiff('rescale1_2_ref_' + fname, rescale(intensityDet, 0.2, mode='constant'))
#save_tiff('rescale1_3_dark_'+ fname, 0.0*rescale(intensitylensdet, 0.2, mode='constant'))
#save_tiff('rescale1_4_grating_'+ fname, rescale(np.angle(gr), 0.2, mode='constant'))

print('Done!')

# %% Save images - rescale and filter

#save_tiff('rescale2_1_image_' + fname,
#                  gaussian_filter(rescale(intensitylensdet, 0.2, mode='constant'), sigma=2))
#save_tiff('rescale2_2_ref_' + fname,
#                  gaussian_filter(rescale(intensityDet, 0.2, mode='constant'), sigma=2))
#save_tiff('rescale2_3_dark_'+ fname,
#                  0.0*rescale(intensitylensdet, 0.2, mode='constant'))
#save_tiff('rescale2_4_grating_'+ fname,
#                  gaussian_filter(rescale(np.angle(gr), 0.2, mode='constant'), sigma=2))
#
#print('Done!')

# %% Save images - rescale mean

#save_tiff('rescale3_1_image_' + fname, downscale_local_mean(intensitylensdet, (5, 5)))
#save_tiff('rescale3_2_ref_' + fname, downscale_local_mean(intensityDet, (5, 5)))
#save_tiff('rescale3_3_dark_'+ fname, 0.0*downscale_local_mean(intensitylensdet, (5, 5)))
#save_tiff('rescale3_4_grating_'+ fname, downscale_local_mean(np.angle(gr), (5, 5)))
#
#print('Done!')


# %%

def simple_reduce(array, factor):
    resized_array = np.zeros((array.shape[0]//factor,
                                         array.shape[1]//factor))

    for i in range(0, factor):
        for j in range(0, factor):

            resized_array += array[i::factor,j::factor]

    return resized_array/factor**2

# %% Save images - rescale my mean

#factor = 4
#
#fname = str(sourceStr + '{:.0f}eV_d_{:.1f}mm_pixel_{:.3f}um_grPeriod_{:.1f}um'.format(phEnergy, zz*1e3, dx*1e6*factor, grPeriod*1e6))
#fname = fname.replace('.', 'p')
#fname += '.tif'
#
#save_tiff('rescale4_1_image_' + fname, simple_reduce(intensitylensdet, factor))
#save_tiff('rescale4_2_ref_' + fname, simple_reduce(intensityDet, factor))
#save_tiff('rescale4_3_dark_'+ fname, 0.0*simple_reduce(intensityDet, factor))

# %%

factor = 4

fname = str(sourceStr + '_{:.0f}eV_d_{:.1f}mm_pixel_{:.3f}um_grPeriod_{:.1f}um'.format(phEnergy, zz*1e3, dx*1e6*factor, grPeriod*1e6))
fname = fname.replace('.', 'p')
fname += '.tif'

save_tiff('rescale5_1_image_' + fname,
          simple_reduce(gaussian_filter(intensitylensdet + 0*noise(intensitylensdet, amp=.01), sigma=10), factor))
save_tiff('rescale5_2_ref_' + fname,
          simple_reduce(gaussian_filter(intensityDet + 0*noise(intensitylensdet, amp=.01), sigma=10), factor))
save_tiff('rescale5_3_dark_'+ fname,
          simple_reduce(intensityDet*0.0 + 0*noise(intensitylensdet, amp=.01), factor))

print('Done!')




# %%

#thickness = - wavelength/2/np.pi/delta * wfLens

#wpu.save_sdf_file(thickness[::10,::10],
#                  [dx*10, dy*10], 'thickness_stride10.sdf')


#bla = thickness[::10,::10]


# %%




