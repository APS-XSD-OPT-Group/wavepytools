# -*- coding: utf-8 -*-  #
"""
Created on %(date)s

@author: %(username)s
"""

# %%% imports cell
import numpy as np
import wavepy.utils as wpu

import matplotlib.pyplot as plt

from wavepy.utils import easyqt
import sys

# %%

def save_profile_from_sdf(fname, nprofiles=3, direction='V', savecsv=True):

    data, pixelsize, headerdic = wpu.load_sdf_file(fname)

    saveFileSuf = fname.replace('.sdf', '')


    if 'V' in direction:

        xvec = wpu.realcoordvec(data.shape[0], pixelsize[0])[np.newaxis]
        data2save = np.c_[xvec.T]

        for j in np.linspace(0, np.shape(data)[1] - 1, nprofiles + 2, dtype=int):
            data2save = np.c_[data2save, data[:, j]]
    else:

        xvec = wpu.realcoordvec(data.shape[1], pixelsize[1])[np.newaxis]
        data2save = np.c_[xvec.T]

        for i in np.linspace(0, np.shape(data)[0] - 1, nprofiles + 2, dtype=int):
            data2save = np.c_[data2save, data[i, :]]


    if savecsv:

        wpu.save_csv_file(data2save,
                          wpu.get_unique_filename(saveFileSuf +
                                                  '_profiles_V', 'csv'),
                                                  headerList='bla')

    return data2save, headerdic




# %%

if __name__ == '__main__':

    argv = sys.argv

    if len(argv) == 4:
        data, headerdic = save_profile_from_sdf(argv[1],
                                     nprofiles=int(argv[2]),
                                     direction=argv[3])

    else:

        data, headerdic = save_profile_from_sdf(easyqt.get_file_names()[0],
                                     nprofiles=easyqt.get_int('Number of profiles'),
                                     direction=easyqt.get_choice(choices=['Vertical', 'Horizontal']))

# %%
    plt.figure()

    for i in range(1, data.shape[1]):
        plt.plot(data[:,0], data[:,i]+1e-9)

    plt.show()

