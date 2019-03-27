#! /bin/bash

SCRIPT='/home/grizolli/workspace/pythonWorkspace/imaging/single_grating/singleCheckerboardGratingTalbot.py'
# FOLDER='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/timescan_turn_all_off_from_all_on/'
FOLDER='/home/grizolli/DATA/20181106_1BM_JTEC/20181109/18_diffractedbeam_linearity_varing_crystal_theta/thetaB_11p7808/'
FPATTERN='*.tif'  # file pattern

for f in $FOLDER$FPATTERN
do
    echo $f
    #     mkdir "${f[@]:22:7}"  
done

while true; do
    read -p "Xianbo, did you crop the file? (so it is saved in the .ini file) [y/n] " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


# DARK_IMG='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/dark_1s_002.tif'
# REF_IMG='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/timescan_turn_all_off_from_all_on/CB4p8pi_84mm_12keV_1s_every5s_000.tif'


DARK_IMG='/home/grizolli/DATA/20181106_1BM_JTEC/20181109/dark_30s_049.tif'

LISTOFFILES=($FOLDER$FPATTERN)

REF_IMG=${LISTOFFILES[0]}


echo 'DARK_IMG: ' $DARK_IMG
echo 'REF_IMG: ' $REF_IMG

while true; do
    read -p "Do you wish to continue? [y/n] " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

paplay /usr/share/sounds/freedesktop/stereo/complete.oga



for f in $FOLDER$FPATTERN
do

    SAMPLE_IMG=$f
    
    #     DIST=$(echo $SAMPLE_IMG | cut -d '_' -f 14 | cut -d 'p' -f 1)

    echo 'Sample'
    echo $SAMPLE_IMG
    echo 'Reference:'
    echo $REF_IMG
    echo 'Dark:'
    echo $DARK_IMG
    
    parallel --bg -j 7 $SCRIPT $SAMPLE_IMG $REF_IMG $DARK_IMG 0.31 4.8 Edge 84.0 12.0 100.0 1 0 0 0 0 0 1
    
    #     parallel --bg -j 7 $SCRIPT $SAMPLE_IMG $REF_IMG $DARK_IMG 0.62 4.8 Diag 56.0 14.0 100.0 1 0 0 0 0 0 1
	
    paplay /usr/share/sounds/freedesktop/stereo/message.oga 

    sleep 2  # avoid some racing when accessing ini file
    
    REF_IMG=$SAMPLE_IMG   
    
done 



parallel  --wait; paplay /usr/share/sounds/freedesktop/stereo/complete.oga; zenity --warning --text='\tDone!\t\t'

cd $FOLDER
mkdir csv_DPC  csv_WF  pngs

mcp '*_output/*/*WF*01.csv' 'csv_DPC/#1.csv';
mcp '*_output/*/*integrated*01.csv' 'csv_WF/#1.csv';
mcp '*/*/*03.*' 'pngs/#1.png'



# 
# arg0: : ./singleCheckerboardGratingTalbot.py
# arg1: file name main image:
# arg2: file name reference image:
# arg2: file name reference image
# arg3: file name dark image
# arg4: pixel size [um]
# arg5: Check Board grating period [um]
# arg6: pattern, 'Edge pi' or 'Diagonal half pi' 
# arg7: distance detector to CB Grating [mm]
# arg8: Photon Energy [KeV]
# arg9: Distance to the source [m]

# arg10: Flag correct pi jump in DPC signal
# arg11: Flag remove mean DPC
# arg12: Flag remove 2D linear fit from DPC
# arg13: Flag Calculate Frankot-Chellappa integration

# arg14: Flag Convert phase to thickness
# arg15: Flag remove 2nd order polynomial from integrated Phase
# arg16: Index for material: 0-Diamond, 1-Be

