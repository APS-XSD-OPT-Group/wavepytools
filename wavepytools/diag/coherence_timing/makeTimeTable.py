# -*- coding: utf-8 -*-  #
"""
Created on Sat Apr  8 12:00:14 2017

@author: grizolli
"""



import numpy as np

# %%
nVec = np.linspace(0, 19, 20)


timeVec = np.round(10*1.13**nVec)/1000


#timeVec = np.repeat(timeVec, 100)


acqVec = timeVec*1.0

#acqVec[np.where(acqVec<.100)] = .1000


#nImages = np.array(1/timeVec, dtype=int)


#nImages[np.where(nImages<10)] = 10


np.savetxt('timeVec.txt', [timeVec], fmt='%.3f')

np.savetxt('acqVec.txt', [acqVec], fmt='%.3f')

#np.savetxt('nImages.txt', [nImages], fmt='%.d')


# %%

repetion = 20
timeVec = np.array([105, 148, 207, 289, 405, 567, 794, 1111])*0.001


timeVec = np.repeat([105, 148, 207, 289, 405, 567], 30)

timeVec = np.concatenate((timeVec,
                          np.repeat([794, 1111], 10)))


timeVec = timeVec*.001

np.savetxt('z616mm_105to1111ms_timeVec.txt', [timeVec], fmt='%.3f')

np.savetxt('z616mm_105to1111ms_nImages.txt', [timeVec*0.0 + 1], fmt='%d')

# %%

repetion = 100
timeVec = np.array([2, 3, 5, 7, 10, 14, 20])*0.001

timeVec = np.repeat(timeVec, repetion)



np.savetxt('z616mm_2to20ms_timeVec.txt', [timeVec], fmt='%.3f')

np.savetxt('z616mm_2to20ms_nImages.txt', [timeVec*0.0 + 1], fmt='%d')



# %%

repetion = 100
timeVec = np.array([14, 20, 27, 38, 54, 75, 105, 148])*0.001

timeVec = np.repeat(timeVec, repetion)



np.savetxt('z616mm_14to148ms_timeVec.txt', [timeVec], fmt='%.3f')

np.savetxt('z616mm_14to148ms_nImages.txt', [timeVec*0.0 + 1], fmt='%d')

# %%


# %%

repetion = 100
timeVec = np.array([22, 25, 30, 33])*0.001

timeVec = np.repeat(timeVec, repetion)



np.savetxt('z616mm_22to33ms_timeVec.txt', [timeVec], fmt='%.3f')

np.savetxt('z616mm_22to33ms_nImages.txt', [timeVec*0.0 + 1], fmt='%d')

