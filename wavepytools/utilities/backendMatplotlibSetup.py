#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
"""
@author: wcgrizolli
"""


# %%
import matplotlib



print('\n\nCurrent Backend: ' + matplotlib.get_backend())
print('\n\nList of all Backend: ')
print(matplotlib.rcsetup.all_backends)

#matplotlib.use('Qt5Agg') #need to be run BEFORE pyplot
print(matplotlib.get_backend())


import matplotlib.pyplot as plt
import numpy as np

plt.ion()
# %%

x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
# %%
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)
# %%
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'yo-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
# %%
plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')
# %%

plt.savefig('1.png')
plt.show(block=True)


print("Press Enter to exit.")
entry=input( )

