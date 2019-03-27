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


from cycler import cycler
import pickle
from wavepy.utils import easyqt
import os
import glob
import matplotlib.pyplot as plt



import numpy as np


plt.rc('axes', prop_cycle=(cycler('color', ['k', 'r', 'g', 'b', 'm', 'r', 'g', 'b', 'm', 'k',
                                            'r', 'g', 'b', 'm', 'r', 'g', 'b', 'm', 'k', 'k']) +
                           cycler('marker', ['o', '.', 'x', 'd', 's', 'o', '.', 'x', 'd', 's',
                                             'o', '.', 'x', 'd', 's', 'o', '.', 'x', 'd', 's'])+
                           cycler('ls', ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--',
                                         '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', ])))

def _mean_for_timing_data(zvec, contrast, label):

    nimages = 10
    new_zvec = np.array([18.0, 56.0, 94.0, 130.0, 166.0, 204.0, 242.0, 278.0,
                     316.0, 352.0, 390.0, 426.0, 462.0, 498.0, 536.0, 572.0,
                     610.0, 644.0, 680.0, 718.0, 756.0, 791.0, 828.0, 864.0])*.001*4.55e-5*1e6

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

    plt.figure("All togethr")

    plt.plot(new_zvec, mean_c, lw=2, label=label)


# %% Load Files

data_dir =  easyqt.get_directory_name("Choose directory with all files")


os.chdir(data_dir)

listOfDataFiles = glob.glob(data_dir+ '/**.pickle')


listOfDataFiles.sort()

#wpu.print_blue('MESSAGE: Loading files ' + \
#                samplefileName.rsplit('_',1)[0] + '*.tif')

#fname_dark =  easyqt.get_file_names("Dark File")[0]


nfiles = len(listOfDataFiles)

# %% Main

#fname = easyqt.get_file_names("Pickle File to Plot")[0]

for fname in listOfDataFiles:


    figx = pickle.load(open(fname,'rb'))
    plt.show(block=False) # this lines keep the script alive to see the plot

    label = fname.rsplit('.')[0].rsplit('/')[-1]
    plt.savefig(label + '.png')
    plt.close()

    curves = []

    for i in range(len(figx.axes[0].lines)):

        curves.append(figx.axes[0].lines[i].get_data())

    curves = np.asarray(curves)

    _mean_for_timing_data(curves[0,0], curves[0,1], label)

plt.xlabel(r'$l_c$ [$\mu m$]', fontsize=20)
plt.ylabel(r'Visbility [a. u.]', fontsize=20)
plt.grid()
plt.legend()
plt.show()

# %% Sandbox



    # %%


#_mean_for_timing_data(curves[0,0], curves[0,1], label)
#
#zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves1[0,0], curves2[0,1], label2)
#
#zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves1[0,0], curves3[0,1], label3)
#
#zvec, mean_cV, min_cV, max_cV = _mean_for_timing_data(curves4[0,0], curves4[0,1], label4)
#
#plt.legend()
#plt.show()
#
#
#
#


# Dump



    ## %%
    #
    #
    #plt.figure()
    #plt.plot(curves1[0,0], curves1[0,1], '-ok')
    #plt.plot(curves2[0,0], curves2[0,1], '-or')
    #plt.plot(curves3[0,0], curves3[0,1], '-ob')
    #plt.plot(curves4[0,0], curves4[0,1], '-og')
    #
    #plt.show()

