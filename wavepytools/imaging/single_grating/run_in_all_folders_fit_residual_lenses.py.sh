#! /bin/bash

SCRIPT='/home/grizolli/workspace/pythonWorkspace/metrology/lenses/fit_residual_lenses.py'

FOLDER='/home/grizolli/workspace/pythonWorkspace/imaging/single_grating/perfect/talbot/'

FPATTERN='T*d0p*/output'



for f in $FOLDER$FPATTERN
do
    echo $f
done


paplay /usr/share/sounds/freedesktop/stereo/complete.oga

for f in $FOLDER$FPATTERN
do

    SAMPLE_IMG=$f/*thickness_01.sdf

    echo 'Sample file:'
    echo $SAMPLE_IMG
    
    parallel --bg -j 7 $SCRIPT $SAMPLE_IMG  'Perfect_Be_Lens' -25.00 0 '150,250'  # WARNING: dont use space inside quote marks
	
    paplay /usr/share/sounds/freedesktop/stereo/message.oga 

    sleep .5  # avoid some racing when accessing ini file
    
done 

parallel  --wait; paplay /usr/share/sounds/freedesktop/stereo/complete.oga; zenity --warning --text='\tDone!\t\t'


# ./fit_residual_lenses.py -h
# ERROR: wrong number of inputs: 1 
# Usage: 
# 
# fit_residual_lenses.py : (no inputs) load dialogs 
# 
# fit_residual_lenses.py [args] 
# 
# arg1: file name with thickness image
# arg2: String for Titles
# arg3: nominal curvature radius for fitting
# arg4: index for lens geometry:
#         0 : 2D Lens Stigmatic Lens
#         1 : 1Dx Horizontal focusing
#         2 : 1Dy Vertical focusing
# arg5: diameter4fit_list:
# 
# 
# arg 0: ./fit_residual_lenses.py
# arg 1: -h

