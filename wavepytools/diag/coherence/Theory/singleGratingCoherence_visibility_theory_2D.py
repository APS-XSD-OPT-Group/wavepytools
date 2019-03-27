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


This Script use the technique described by Xianbo Shi in
https://doi.org/10.1364/OE.22.014041

'''



#There is a problem in the scalling of z for tilted grating


# %%
import numpy as np

import matplotlib as mpl
mpl.rcParams['image.interpolation']='none'

import matplotlib.pyplot as plt

import itertools

import scipy.constants as constants

import skimage.transform

hc = constants.value('inverse meter-electron volt relationship')  # hc


def _checkerboard(shape,transmission=1.0, phase=np.pi):


    checkerboard = np.ones(shape)*0j
    checkerboard[0:shape[0] // 2,
                0:shape[1] // 2] = transmission*np.exp(1j*phase)


    checkerboard[shape[0] // 2:shape[0],
                 shape[1] // 2:shape[1]] = transmission*np.exp(1j*phase)

    return checkerboard


def _mesh(shape, transmission=1.0, phase=np.pi, inverseDutyCycle=2):


    mesh = np.ones(shape)*0j
    mesh[0:shape[0] // inverseDutyCycle,
         0:shape[1] // inverseDutyCycle] = transmission*np.exp(1j*phase)


    return mesh


# %% create grating

periodX = periodY = 4.8e-6
Lx = Ly = periodX

phase_gr = np.pi/2  # TODO:

global wavelength
wavelength = hc/8e3





npoints = 100
if npoints % 2 == 0:
    npoints += 1



yy, xx = np.mgrid[0:Lx:npoints*1j, 0:Ly:npoints*1j]
grProfile = _checkerboard(xx.shape,
                           transmission=1.00,
                           phase=phase_gr)




#grProfile = _mesh(xx.shape,
#                  transmission=1.00,
#                  phase=phase_gr,
#                  inverseDutyCycle=2)


# rotate CB 45 deg
#grProfile = np.concatenate((grProfile, grProfile), axis=0)
#grProfile = np.concatenate((grProfile, grProfile), axis=1)
#grProfile.real  = skimage.transform.rotate(grProfile.real, 45, mode='wrap')
#grProfile.imag = skimage.transform.rotate(grProfile.imag, 45, mode='wrap')
#
#
#grProfile = np.roll(np.roll(grProfile, 20, axis=1), 20, axis=0)
#
#grProfile = grProfile[int(npoints*(1-np.sqrt(2)/4)):int(npoints*(1+np.sqrt(2)/4)),
#                      int(npoints*(1-np.sqrt(2)/4)):int(npoints*(1+np.sqrt(2)/4))]
#
#periodX = periodY = 4.8e-6*np.sqrt(2)/2
#Lx = Ly = periodX
#
#yy, xx = np.mgrid[0:Lx:npoints*1j, 0:Ly:npoints*1j]


#
#grProfile = np.concatenate((grProfile, grProfile), axis=0)
#grProfile = np.concatenate((grProfile, grProfile), axis=1)
#
#Lx = Ly = 4*periodX
#yy, xx = np.mgrid[0:Lx:npoints*1j, 0:Ly:npoints*1j]



t_distance = 2*periodX**2/wavelength
dist4all = t_distance/2  # TODO:


titleStr = 'CB, {:.2f}'.format(phase_gr/np.pi) + r'$\times \pi$, '

# %% rebininb detector

#
#import scipy.ndimage
#
#
#grProfile_average_i = scipy.ndimage.uniform_filter(np.imag(grProfile), size=12,
#                                                   output=None, mode='wrap',
#                                                   origin=0)
#
#grProfile_average_r = scipy.ndimage.uniform_filter(np.real(grProfile), size=12,
#                                                   output=None, mode='wrap',
#                                                   origin=0)*144
#
#grProfile = grProfile_average_r[::12,::12] + 1j*grProfile_average_i[::12,::12]
#
#npoints = grProfile.shape[0]
#
#yy, xx = np.mgrid[0:Lx:npoints*1j, 0:Ly:npoints*1j]


# %% plot grating

def _extent(xx, yy, multFactor):
    return [xx[0, 0]*multFactor, xx[-1, -1]*multFactor,
            yy[0, 0]*multFactor, yy[-1, -1]*multFactor]



fig = plt.figure()
ax1 = plt.subplot(121)
ax1.imshow(np.real(grProfile), vmax=1, vmin=-1, extent=_extent(xx, yy, 1/periodX))
ax1.set_title(titleStr + 'Real part')

ax2 = plt.subplot(122)
ax2.imshow(np.imag(grProfile), vmax=1, vmin=-1, extent=_extent(xx, yy, 1/periodX))
ax2.set_title(titleStr + 'Imaginary part')

plt.show(block=True)



# %% Fourier Optics propagation


import sys
sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools')
import myFourierLib as wgfo



grProfile2 = grProfile

dist4fop = dist4all

u2_Summerfield = wgfo.propTF_RayleighSommerfeld(grProfile2,
                                           xx[-1, -1] - xx[0,0],
                                           yy[-1, -1] - yy[0,0],
                                           wavelength, dist4fop)


# %% plot Fourier Optics propagation
plt.figure()
plt.imshow(np.abs(u2_Summerfield), cmap='Greys_r',
           extent=_extent(xx, yy, 1e6))
plt.title(titleStr + 'Fourier Optics Result, d={:.2f}mm'.format(dist4fop*1e3))
plt.xlabel(r'x [$\mu m$]')
plt.ylabel(r'y [$\mu m$]')
plt.show(block=False)

#plt.figure()
#plt.imshow(np.real(u2_Summerfield), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + 'Real Fourier Optics Result, d={:.2f}mm'.format(dist4fop*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.show(block=False)
#
#
#plt.figure()
#plt.imshow(np.imag(u2_Summerfield), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + 'Imag Fourier Optics Result, d={:.2f}mm'.format(dist4fop*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.show(block=False)
#
#
#plt.figure()
#plt.imshow(np.angle(u2_Summerfield), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + 'Angle Fourier Optics Result, d={:.2f}mm'.format(dist4fop*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.colorbar()
#plt.show(block=False)

#
#
#plt.figure()
#plt.plot(np.abs(u2_Summerfield**2)[35,:], '-ok', label='abs**2')
#plt.plot(np.real(u2_Summerfield)[35,:], '-or', label='real')
#plt.plot(np.imag(u2_Summerfield)[35,:], '-ob', label='imag')
#plt.legend()
#plt.title(titleStr + 'Fourier Optics Result, d={:.2f}mm'.format(dist4fop*1e3))
#plt.show(block=True)


# %%
#exit()

# %% def amn

def _amn(grProfile):

    npointsX, npointsY = np.shape(grProfile)
    return np.fft.fft2(grProfile)/npointsX/npointsY


# %% amn

amn = _amn(grProfile)

#an2 = np.abs(np.fft.fftshift(an))
amn2 = np.abs(amn)

npointsX, npointsY = np.shape(amn)

freq_vecX = np.array(np.fft.fftfreq(npointsX, 1/npointsX))
freq_vecY = np.array(np.fft.fftfreq(npointsY, 1/npointsY))

freqX, freqY = np.meshgrid(freq_vecX, freq_vecY, indexing='ij')


# %% plot amn
fig = plt.figure()
plt.imshow(np.log10(np.fft.fftshift(amn2) + 1), cmap='plasma')
plt.title(titleStr + r'Log $|a_{mn}| $')
plt.show(block=True)


# %% def bmn and cuv

def _bmn(dist):

    return amn*np.exp(-1j*np.pi*wavelength*dist*(
                      freqX**2/Lx**2 + freqY**2/Ly**2))



#def _cuv(u, v, bmn):
#
#        Bmn = np.roll(np.roll(bmn, u, axis=0), v, axis=1)
#
#        return np.sum(bmn*np.conj(Bmn))


def _cuv(u, v, amn, dist):

        Amn = np.roll(np.roll(amn, -u, axis=0), -v, axis=1)

        Euv = np.exp(-1j*np.pi*wavelength*dist*(
                      (u**2-2*u*freqX)/Lx**2 + (v**2-2*v*freqY)/Ly**2))

        return np.sum(amn*np.conj(Amn)*Euv)




# %% bmn


bmn = _bmn(dist4all)

# %% u2 from harmonics

nharm_4u2 = amn.shape[0] // 2
if nharm_4u2 > 10:
    nharm_4u2=10
nharm_4u2_vec = np.mgrid[- nharm_4u2:nharm_4u2 + 1]


u2 = xx*0j

for m, n in itertools.product(nharm_4u2_vec, nharm_4u2_vec):
    print("m, n: {}, {}".format(m, n))


    u2 += bmn[m, n]*np.exp(-2j*np.pi*(m*xx/Lx + n*yy/Ly))


# %% plot intensity from harmonics
plt.figure()
plt.imshow(np.abs(u2**2), cmap='Greys_r')
plt.title(titleStr + r'$|U_2|$ from $b_{mn}$,' +
          ' using {} harmonics, d={:.2f}mm,'.format(nharm_4u2, dist4all*1e3))
plt.show(block=False)

#
#plt.figure()
#plt.imshow(np.real(u2), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + r'Real $U_2$ from $b_{mn}$,' +
#          ' using {} harmonics, d={:.2f}mm,'.format(nharm_4u2, dist4all*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.show(block=False)
#
#
#plt.figure()
#plt.imshow(np.imag(u2), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + r'Imag $U_2$ from $b_{mn}$,' +
#          ' using {} harmonics, d={:.2f}mm,'.format(nharm_4u2, dist4all*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.show(block=False)
#
#
#plt.figure()
#plt.imshow(np.angle(u2), cmap='Greys_r',
#           extent=_extent(xx, yy, 1e6))
#plt.title(titleStr + r'Angle $U_2$ from $b_{mn}$,' +
#          ' using {} harmonics, d={:.2f}mm,'.format(nharm_4u2, dist4all*1e3))
#plt.xlabel(r'x [$\mu m$]')
#plt.ylabel(r'y [$\mu m$]')
#plt.colorbar()
#plt.show(block=False)
#
##
#
#plt.figure()
#plt.plot(np.abs(u2**2)[35,:], '-ok', label='abs**2')
#plt.plot(np.real(u2)[35,:], '-or', label='real')
#plt.plot(np.imag(u2)[35,:], '-ob', label='imag')
#plt.legend()
#plt.title(titleStr + '$U_2$ from $b_{mn}$,' +
#          ' using {} harmonics, d={:.2f}mm,'.format(nharm_4u2, dist4all*1e3))
#plt.show(block=True)


# %% cmn dist

dist_vec = np.linspace(0, t_distance, 256 +1)

c00 = []

for d in dist_vec:
    c00.append(_cuv(0, 0, amn, d))

c00 = np.array(c00)

cuv_list =[]


harmonics_to_plot = [[0,0], [0,1], [1,1], [0,2], [2,1], [2,2], [3,1]]

#harmonics_to_plot = [[0,0], [2,0], [2,1], [2,2], [3,1], [4, 1], [4, 2], [4, 3], [4, 4]]

for u, v in harmonics_to_plot:

    print("u, v: {}, {}".format(u, v))
    cuv = []

    #    for d in dist_vec:
    #        cuv.append(_cuv(u, v, _bmn(d)))

    for i in range(dist_vec.shape[0]):
        cuv.append(_cuv(u, v, amn, dist_vec[i]))


    cuv_list.append(cuv)


# %% plot cmn dist


c00 = np.array(cuv_list[0])

plt.figure()


for i in range(1, len(cuv_list)):


    label = str('{}, {}'.format(harmonics_to_plot[i][0],
                                harmonics_to_plot[i][1]))

    plt.plot(dist_vec*1e3, np.abs(np.array(cuv_list[i]))/np.abs(c00),
             '-o', label=label)


plt.title(titleStr + r'$|c_{mn}|$')
plt.legend(title='Normalized by 00')
plt.show(block=False)







# %% bmn dist

bm_dist = np.empty((freqX.shape[0], freqX.shape[1], dist_vec.shape[0]), dtype=complex)

for i in range(dist_vec.shape[0]):


    bm_dist[:,:,i] = _bmn(dist_vec[i])


# %% plot bmn dist

plt.figure()
for m, n in [[0, 1], [1, 1], [2, 0], [2, 1], [2, 2]]:


    plt.plot(dist_vec/t_distance, np.angle(bm_dist[m, n, :])/np.pi,
             '-o', label='{}, {}'.format(m, n))


plt.axhline(np.array(-.5), lw=2)
plt.axhline(np.array(0.0), lw=2)
plt.axhline(np.array(0.5), lw=2)
plt.title(titleStr + r'phase $b_{mn}$')
plt.grid()
plt.legend()
plt.show(block=True)



# %%

exit()

# %% def cmn matrix


def _cuv_matrix(nharm_u, nharm_v, dist):

    uu, vv = np.indices((nharm_u*2 + 1, nharm_v*2 + 1))

    uu -=  nharm_u
    vv -= nharm_v

    cuv = uu*0j

    bmn = _bmn(dist)

    for u in uu.flatten():
        print('Hi u:{}'.format(u))

        Bmn_shift_u = np.roll(bmn, -u, axis=0)

        for v in vv.flatten():

            Bmn = np.roll(Bmn_shift_u, -v, axis=1)

            cuv[u, v] = np.sum(bmn*np.conj(Bmn))


    return cuv



# %% cmn matrix


#dist4all = t_distance*1/16
cuv = _cuv_matrix(10, 10, dist4all)

#cuv2 = np.abs(cuv)**2
cuv2 = np.abs(np.fft.fftshift(cuv)**2)

# %% plot cmn matrix

plt.figure()
plt.imshow(cuv2)

plt.title(titleStr + r'$|c_{mn}|,$' + ' d={:.2f}mm'.format(dist4all*1e3))
#plt.imshow(np.log10(cuv2) cmap='RdGy')
plt.colorbar()
plt.show(block=True)




# %% intensity from cmn

nharm_cmn = np.shape(cuv)[0]
nharm_cmn_vec = np.mgrid[-10:10+1]

intensity = xx*0j

for m, n in itertools.product(nharm_cmn_vec, nharm_cmn_vec):

    print("m, n: {}, {}".format(m, n))
    intensity += cuv[m, n]*np.exp(-2j*np.pi*(
                    m*xx/Lx + n*yy/Ly))

# %%





plt.figure()
plt.imshow(np.abs(intensity), cmap='Greys_r')
plt.title(titleStr + r'Intensity from $|c_{mn}|$,' + ' d={:.2f}mm'.format(dist4all*1e3))
plt.show(block=True)





