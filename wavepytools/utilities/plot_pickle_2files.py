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
import matplotlib.pyplot as plt

figx = pickle.load(open(easyqt.get_file_names("Pickle File to Plot")[0],'rb'))

plt.show(block=False) # this lines keep the script alive to see the plot


figx2 = pickle.load(open(easyqt.get_file_names("Pickle File to Plot")[0],'rb'))

plt.show(block=True) # this lines keep the script alive to see the plot


# %% Example of how to get the data from a graph



import numpy as np


curves1 = []

for i in range(len(figx.axes[0].lines)):

    curves1.append(figx.axes[0].lines[i].get_data())

curves1 = np.asarray(curves1)


curves2 = []

for i in range(len(figx2.axes[0].lines)):

    curves2.append(figx2.axes[0].lines[i].get_data())

curves2 = np.asarray(curves2)

# %% Sandbox
import wavepy.utils as wpu
wpu._mpl_settings_4_nice_graphs(fs=20)


plt.figure(figsize=(10,8))

plt.plot(curves1[0,0,:], curves1[0,1,:]/np.max(curves1[0,1,:]), '-ok', label= 'Direct beam')
plt.plot(curves2[0,0,:], curves2[0,1,:]/np.max(curves2[0,1,:]), '-og', label= 'Reflected beam')

plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Visibility')

plt.legend()
plt.show()


# %%


plt.figure(figsize=(10,8))

plt.plot(curves1[1,0,:], curves1[1,1,:]/np.max(curves1[0,1,:]), '-ok', label= 'Direct beam')
plt.plot(curves2[1,0,:], curves2[1,1,:]/np.max(curves2[0,1,:]), '-og', label= 'Reflected beam')

plt.xlabel(r'Distance [mm]')
plt.ylabel(r'Visibility')

plt.legend()
plt.show()




