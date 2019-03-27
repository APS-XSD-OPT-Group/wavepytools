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

import numpy as np


import matplotlib.pyplot as plt

import scipy.constants as constants

hc = constants.value('inverse meter-electron volt relationship')  # hc

#import wavepy.utils as wpu
#import wavepy.grating_interferometry as wgi
#import wavepy.surface_from_grad as wps


def _square_wave(xvec, period,
                 transmission=1.0,
                 phase=np.pi,
                 duty_cycle=0.500):


    xvec = np.mgrid[0:period:npoints*1j]
    square_wave = np.ones(np.shape(xvec))*0j
    square_wave[0:int(duty_cycle*npoints)] = transmission*np.exp(1j*phase)

    return square_wave


# %%

period = 4.8e-6

phase_gr = np.pi

global wavelength

wavelength = hc/8e3/2

npoints = 1000
xvec = np.mgrid[0:period:npoints*1j]
dx = xvec[1] - xvec[0]
grProfile = _square_wave(xvec, period,
                         transmission=1.00,
                         phase=phase_gr, duty_cycle = 0.25)

#grProfile = np.cos(2*np.pi*xvec/period + np.pi/3) + .5*np.cos(2*2*np.pi*xvec/period)

# %%

def _an(grProfile):

    npoints = np.shape(grProfile)[0]
    return np.fft.fft(grProfile)/npoints


def _cn(n_, distance, an, period):

    npoints = np.shape(grProfile)[0]

    freq_vec = np.array(np.fft.fftfreq(npoints, 1/npoints))

    bn = an*(0.0 + 1j*0.0)

    for n in range(npoints):
        bn[n] = an[n]*np.exp(-1j*np.pi*wavelength*distance*freq_vec[n]**2/period**2)

    cn = 0.0 + 0.0*1j

    bn = np.concatenate((bn, bn, bn))

    for m in range(npoints):

        cn += bn[n_ + m]*np.conj(bn[m])

    return cn

# %%

fig = plt.figure()
plt.plot(xvec*1e6, np.real(grProfile), '-ko', label='Real part')
plt.plot(xvec*1e6, np.imag(grProfile), '-ro', label='Imaginary part')
plt.grid()
plt.legend()
plt.title(r'Grating Profile')
plt.show()

# %%
an = _an(grProfile)

#an2 = np.abs(np.fft.fftshift(an))
an2 = np.abs(an)

freq_vec = np.array(np.fft.fftfreq(npoints, 1/npoints))


# %%
fig = plt.figure()
plt.plot(np.fft.fftshift(freq_vec), np.fft.fftshift(an2), '-ko')
plt.grid()
plt.title(r'$a_n^2$')
plt.show()



# %%


nharm_cn = 100
nharm_cn_vec = np.mgrid[-nharm_cn:nharm_cn + 1]


t_distance = 2*period**2/wavelength

dist = t_distance/4 # t_distance
cn = [_cn(n, dist, an, period) for n in nharm_cn_vec]


fig = plt.figure()
plt.plot(nharm_cn_vec, np.abs(cn), '-ko')
plt.grid()
plt.title(r'$| c_n| $')
plt.show()

#


intensity = xvec*(0.0 +1j*0.0)

fig = plt.figure()

for n in range(np.size(cn)):

    intensity += cn[n]*np.exp(2j*np.pi*nharm_cn_vec[n]*xvec/period)

    if nharm_cn_vec[n] > -1:
        plt.plot(xvec*1e6,
                 np.real(cn[n]*np.exp(2j*np.pi*nharm_cn_vec[n]*xvec/period)),
                         label=str(nharm_cn_vec[n]))

plt.legend()
plt.grid()
plt.title(r'Harmonics')
plt.show()

fig = plt.figure()
plt.plot(xvec*1e6, np.real(intensity), '-ko', label='Real part')
plt.plot(xvec*1e6, np.imag(intensity), '-r', label='Imaginary part')
plt.grid()
plt.title(r'Intensity at distance={:.1f}m'.format(dist*1e3))
plt.legend()
plt.show()


# %%

zstep = .005
max_dist = 1.50  # 3.1*t_distance
dist_vec = np.arange(.000, max_dist, zstep)

cn0_dist = [np.abs(_cn(0, dist, an, period)) for dist in dist_vec]

c0_max = np.max(cn0_dist)

fig = plt.figure()

#plt.plot(dist_vec*1e3, cn0_dist/c0_max, '-o', label='c{}'.format(0))

plt.plot(dist_vec/t_distance, cn0_dist/c0_max, '-o', label='c{}'.format(0))

for n_to_plot in range(1, 8):

    dist_vec = np.arange(.000, max_dist, zstep/n_to_plot)

    cn_dist = [np.abs(_cn(n_to_plot, dist, an, period)) for dist in dist_vec]

    #    plt.plot(dist_vec*1e3, cn_dist/c0_max, '-o', label='{}'.format(n_to_plot))
    plt.plot(dist_vec/t_distance, cn_dist/c0_max, '-o', label='{}'.format(n_to_plot))

#text_y_position = plt.gca().get_ylim()[1]
#for n_talbot in np.arange(1, dist_vec[-1] // t_distance + 1, dtype=int):
#
#    plt.axvline(n_talbot*t_distance*1e3, ls='--', lw=2, color='k')
#
#    plt.annotate(r'$T{:d}$'.format(n_talbot),
#                 (n_talbot*t_distance*1e3, text_y_position),
#                 xytext=(10, -20), textcoords='offset points',
#                 bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.95),
#                 fontsize=18, weight='bold')


plt.title(r'$c_n$ vs distance', fontsize=14, weight='bold')
plt.grid()
# plt.ylim([0, 1.2])
plt.legend(fontsize=14)
plt.show()



# %%

dcyvle = np.array([.1, .25, .5, .75, .9])

c0_dcycle=[]
c1_dcycle=[]

for dcyvle_val in dcyvle:

    grProfile_dcyvle = _square_wave(xvec, period,
                                    transmission=1.00,
                                    phase=phase_gr, duty_cycle= dcyvle_val)


    an_dcyvle = _an(grProfile_dcyvle)
    c0_dcycle.append( _cn(0, t_distance, an_dcyvle, period))
    c1_dcycle.append( _cn(1, t_distance, an_dcyvle, period))


c0_dcycle = np.abs(np.array(c0_dcycle))
c1_dcycle = np.abs(np.array(c1_dcycle))

# %%

plt.figure()
plt.plot(dcyvle, c0_dcycle, '-o', label=r'$c_0$')
plt.plot(dcyvle, c1_dcycle, '-o', label=r'$c_1$')
plt.plot(dcyvle, c1_dcycle/c0_dcycle, '-o', label=r'$c_1/c_0$')
plt.legend()
plt.show()


# %%

