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


#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

import wavepy.utils as wpu



def _load_data_from_pickle(fname):

    fig = pickle.load(open(fname,'rb') )

    fig.set_size_inches((12,9), forward=True)


    plt.show(block=True) # this lines keep the script alive to see the plot


    curves = []

    for i in range(len(fig.axes[0].lines)):

        curves.append(np.asarray(fig.axes[0].lines[i].get_data()))


    return curves


# %%

if __name__ == '__main__':

    #    if len(sys.argv) != 1:
    #        os.chdir(sys.argv[1])


    if len(sys.argv) == 1:

        flist = easyqt.get_file_names("Pickle File to Plot")

        fname = flist[0]

    else:

       fname = sys.argv[1]



    results = _load_data_from_pickle(fname)


    # %%
    zvec = results[0][0]*1e-3
    contrastV = results[0][1]*1e-2
    contrastH = results[1][1]*1e-2

    if zvec[-1] - zvec[0] < 0:
        zvec = zvec[::-1]
        contrastV = contrastV[::-1]
        contrastH = contrastH[::-1]


    # %%
    new_zvec = np.linspace(0, 23, 24)

    new_zvec = np.array([18.0, 56.0, 94.0, 130.0, 166.0, 204.0, 242.0, 278.0,
                         316.0, 352.0, 390.0, 426.0, 462.0, 498.0, 536.0, 572.0,
                         610.0, 644.0, 680.0, 718.0, 756.0, 791.0, 828.0, 864.0])

    mean_cV = new_zvec*0.0
    mean_cH = new_zvec*0.0

    max_cV = new_zvec*0.0
    max_cH = new_zvec*0.0

    min_cV = new_zvec*0.0
    min_cH = new_zvec*0.0


    nimages = 10

    for n in range(zvec.size // nimages):

        zvec[n*nimages:(n+1)*nimages] = new_zvec[n]


        mean_cV[n] = np.mean(contrastV[n*nimages:(n+1)*nimages])
        mean_cH[n] = np.mean(contrastH[n*nimages:(n+1)*nimages])

        max_cV[n] = np.max(contrastV[n*nimages:(n+1)*nimages])
        max_cH[n] = np.max(contrastH[n*nimages:(n+1)*nimages])


        min_cV[n] = np.min(contrastV[n*nimages:(n+1)*nimages])
        min_cH[n] = np.min(contrastH[n*nimages:(n+1)*nimages])

    # %%

    mean_cV /= np.max(mean_cV)
    mean_cH /= np.max(mean_cH)



    # %%

    plt.figure(figsize=(12,9))
    plt.plot(new_zvec, mean_cV, '-ko', lw=2)
    plt.plot(new_zvec, mean_cH, '-rs', lw=2)



    plt.plot(new_zvec, max_cV, '--k', lw=2)
    plt.plot(new_zvec, max_cH, '--r', lw=2)


    plt.plot(new_zvec, min_cV, '--k', lw=2)
    plt.plot(new_zvec, min_cH, '--r', lw=2)

    plt.plot(zvec, contrastV, 'ok')
    plt.plot(zvec, contrastH, 'sr')

    plt.xlabel(r'z distance [mm]')
    plt.title(fname.rsplit('.')[0].rsplit('/')[-1])


    wpu.save_figs_with_idx(fname.rsplit('.')[0].rsplit('/')[-1])

    plt.show()








