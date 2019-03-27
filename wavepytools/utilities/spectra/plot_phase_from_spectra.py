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

"""
@author: Walan Grizolli

Hint: make the file executable and add to a directory in the PATH

to make executable in linux:

>>> chmod +x FILENAME

"""


import pickle
from wavepy.utils import easyqt
import os
import sys
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase


import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) != 1:
    os.chdir(sys.argv[1])


os.chdir('/home/grizolli/work/spectra/tempresults/')

data = np.loadtxt(open(easyqt.get_file_names("Data File to Plot")[0], 'rb'),
                  skiprows=11)

# %%

nx = len(set(data[:, 0]))
ny = len(set(data[:, 1]))

xxGrid = np.reshape(data[:, 0]*1e-3, (nx, ny))
yyGrid = np.reshape(data[:, 1]*1e-3, (nx, ny))

field_x = np.reshape(data[:, 2] + 1j*data[:, 3], (nx, ny))
field_y = np.reshape(data[:, 4] + 1j*data[:, 5], (nx, ny))

phase_x = -unwrap_phase(np.angle(field_x))

phase_x -= np.min(phase_x)

plt.figure()
plt.imshow(phase_x)
plt.show(block=True)  # this lines keep the script alive to see the plot

# %%


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(xxGrid*1e6, yyGrid*1e6,  phase_x/2/np.pi,
                       cmap='viridis', linewidth=0.1)

plt.xlabel('[um]')
plt.ylabel('[um]')

plt.title(r'Spectra: Phase / 2 $\pi$ ', fontsize=18, weight='bold')
plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()

