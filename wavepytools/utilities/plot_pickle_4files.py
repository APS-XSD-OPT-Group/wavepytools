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

fname = easyqt.get_file_names("Pickle File to Plot")[0]
figx = pickle.load(open(fname,'rb'))
plt.show(block=False) # this lines keep the script alive to see the plot
label1 = fname.rsplit('.')[0].rsplit('/')[-1]

fname = easyqt.get_file_names("Pickle File to Plot")[0]
figx2 = pickle.load(open(fname,'rb'))
plt.show(block=False) # this lines keep the script alive to see the plot
label2 = fname.rsplit('.')[0].rsplit('/')[-1]

fname = easyqt.get_file_names("Pickle File to Plot")[0]
figx3 = pickle.load(open(fname,'rb'))
plt.show(block=False) # this lines keep the script alive to see the plot
label3 = fname.rsplit('.')[0].rsplit('/')[-1]

fname = easyqt.get_file_names("Pickle File to Plot")[0]
figx4 = pickle.load(open(fname,'rb'))
plt.show(block=False) # this lines keep the script alive to see the plot
label4 = fname.rsplit('.')[0].rsplit('/')[-1]

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



curves3 = []

for i in range(len(figx3.axes[0].lines)):

    curves3.append(figx3.axes[0].lines[i].get_data())

curves3 = np.asarray(curves3)



curves4 = []

for i in range(len(figx4.axes[0].lines)):

    curves4.append(figx4.axes[0].lines[i].get_data())

curves4 = np.asarray(curves4)

# %%


plt.figure()
plt.plot(curves1[0,0], curves1[0,1], '-ok')
plt.plot(curves2[0,0], curves2[0,1], '-or')
plt.plot(curves3[0,0], curves3[0,1], '-ob')
plt.plot(curves4[0,0], curves4[0,1], '-og')

plt.show()


# %% Sandbox

def _mean_for_timing_data(zvec, contrast, label):

    nimages = 10
    new_zvec = np.array([18.0, 56.0, 94.0, 130.0, 166.0, 204.0, 242.0, 278.0,
                     316.0, 352.0, 390.0, 426.0, 462.0, 498.0, 536.0, 572.0,
                     610.0, 644.0, 680.0, 718.0, 756.0, 791.0, 828.0, 864.0])

    mean_c = new_zvec*0.0
    min_c = new_zvec*0.0
    max_c = new_zvec*0.0

    for n in range(zvec.size // nimages):

        zvec[n*nimages:(n+1)*nimages] = new_zvec[n]


        mean_c[n] = np.mean(contrast[n*nimages:(n+1)*nimages])

        max_c[n] = np.max(contrast[n*nimages:(n+1)*nimages])


        min_c[n] = np.min(contrast[n*nimages:(n+1)*nimages])

    norm = 1/np.max(mean_c)

    mean_c *= norm
    max_c *= norm
    min_c *= norm

    plt.plot(new_zvec, mean_c, '-o', label=label)

    return new_zvec, mean_c*norm, max_c*norm, min_c*norm

    # %%

plt.figure()
zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves1[0,0], curves1[0,1], label1)

zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves1[0,0], curves2[0,1], label2)

zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves1[0,0], curves3[0,1], label3)

zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves4[0,0], curves4[0,1], label4)

plt.legend()
plt.show()






