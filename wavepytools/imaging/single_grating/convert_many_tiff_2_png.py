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



'''
# %%


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import wavepy.utils as wpu
from wavepy.utils import easyqt

import dxchange
import sys

import os, glob

# %% Load


if __name__ == '__main__':

    if len(sys.argv) == 1:

        samplefileName = easyqt.get_file_names("Choose the reference file " +
                                               "for alignment")[0]

    else:
        samplefileName = sys.argv[1]


    if '/' in samplefileName:
        data_dir = samplefileName.rsplit('/', 1)[0]
        os.chdir(data_dir)
    else:
        data_dir = os.getcwd()


    os.makedirs('png_figs', exist_ok=True)

    wpu.print_blue('MESSAGE: Loading files ' +
               samplefileName.rsplit('_', 1)[0] + '*.tif')

    listOfDataFiles = glob.glob(data_dir + '/*.tif')
    listOfDataFiles.sort()


    # %% Loop over the files in the folder

    for i, imgfname in enumerate(listOfDataFiles):
        if i == 0:

            img_0 = dxchange.read_tiff(imgfname)
            all_img = np.zeros((len(listOfDataFiles),
                                img_0.shape[0], img_0.shape[1]), dtype=int)

        all_img[i,:,:] = dxchange.read_tiff(imgfname)

        wpu.print_blue('MESSAGE: Loading ' + imgfname)

    for i, imgfname in enumerate(listOfDataFiles):

        plt.figure()
        plt.figure(figsize=(12, 12*9/16))

        plt.imshow(all_img[i, :, :],
                   cmap='jet',
                   vmin=0, vmax=wpu.mean_plus_n_sigma(all_img, 6)//1)

        outfname = 'png_figs/' + \
                   imgfname.split('.')[0].rsplit('/', 1)[-1] + '.png'
        plt.title(imgfname.split('/')[-1])
        plt.savefig(outfname)
        print(outfname + ' saved!')

        plt.close()

#        if i > 0: break

    # %%

#    wpu.print_red('MESSAGE: Done!')
    #    plt.show(block=True)

# %%


