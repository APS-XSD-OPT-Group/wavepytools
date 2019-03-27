# -*- coding: utf-8 -*-  #
"""
Created on Fri Mar 20 16:51:27 2015

@author: wcgrizolli
"""

import numpy as np
from scipy.special import fresnel
import matplotlib.pyplot as plt

t = np.linspace(0, 50.0, 2001)
ss, cc = fresnel(t / np.sqrt(np.pi / 2))
scaled_ss = np.sqrt(np.pi / 2) * ss
scaled_cc = np.sqrt(np.pi / 2) * cc
plt.plot(t, scaled_cc, 'g--', t, scaled_ss, 'r-', linewidth=2)
plt.grid(True)
plt.show()
