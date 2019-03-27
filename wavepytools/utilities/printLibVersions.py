#!/usr/bin/env python
# -*- coding: utf-8 -*-  #

import sys
import scipy
import matplotlib
import numpy
#import sympy
import h5py
#import mpi4py

import os
print(os.getcwd())

print('\nsys.version:\n' + str(sys.version))
print('\nscipy.__version__: ' + str(scipy.__version__))
print('\nnumpy.__version__: ' + str(numpy.__version__))
#print('\nsympy.__version__: ' + str(sympy.__version__))
print('\nmatplotlib.__version__: ' + str(matplotlib.__version__))
print('\nmatplotlib.get_backend(): ' + str(matplotlib.get_backend()))
print('\nh5py.__version__: ' + str(h5py.__version__))
#print('\nmpi4py.__version__: ' + str(mpi4py.__version__))

print("\n\n")


print('Press ENTER to finish')
print(input())

