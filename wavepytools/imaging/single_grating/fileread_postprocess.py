'''
    find the 1D line in the integrated phase in the csv file after wavepy processing
'''

import os
import glob
import numpy as np
import csv
from matplotlib import pylab as plt
import scipy.io as sio

# Folder name
Folder_path = 'C:/Users/zqiao/Documents/xray_wavefront/Experiment_CB4p8halfpi_170mm_20s/R-0000/'
Result_path = 'C:/Users/zqiao/Documents/xray_wavefront/Experiment_CB4p8halfpi_170mm_20s/R-0000/wavefront/'
file_name = '*_integrated_y_01.csv'

if not os.path.exists(Result_path):
    os.makedirs(Result_path)

os.chdir(Folder_path)
print(os.getcwd())
f_list = glob.glob('*_output/profiles/*_integrated_y_01.csv')


x_pos = []
y_pos = []
dataname = []

for kk, f_single in enumerate(f_list):
    # if os.path.split(f_single)[0][0:11] == 'E-0003-0015':
    #     continue
    with open(f_single) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        skipline = 1
        x = []
        y = []
        for row in readCSV:
            if skipline > 0:
                skipline -= 1
                continue
            else:
                x.append(float(row[0]))
                y.append(float(row[2]))
    x_pos.append(x)
    y_pos.append(y)

    dataname.append(os.path.split(f_single)[0][0:6])
    print(dataname[-1])    
    print((kk+1)/len(f_list))

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
dataname = np.array(dataname)

sio.savemat(Result_path+'wavefront.mat', {'dataname': dataname, 'x_axis': x_pos, 'y_axis': y_pos})
# kk = 0
for x, y in zip(x_pos*1e3, y_pos*1e9):
    plt.plot(x, y)
    # plt.pause(1)
    # plt.show(block=False)
    # print(dataname[kk])
    # kk += 1
plt.xlabel('x (mm)')
plt.ylabel('y (nm)')
plt.legend(dataname)
plt.savefig(Result_path+'wavefront.png', transparent=True)
plt.show()



