#! /bin/bash

SCRIPT='/home/grizolli/workspace/pythonWorkspace/imaging/single_grating/singleCheckerboardGratingTalbot.py'
# FOLDER='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/timescan_turn_all_off_from_all_on/'
FOLDER='/home/grizolli/DATA/20181106_1BM_JTEC/20181108/02_direct_beam_response_at_all500/aligned_tiff/'
FPATTERN='dir_9'

for f in $FOLDER$FPATTERN
do
    echo $f
    #     mkdir "${f[@]:22:7}"  
done

# exit

paplay /usr/share/sounds/freedesktop/stereo/complete.oga


DARK_IMG='/home/grizolli/DATA/20181019_RealMirror/20181019/dark_500ms.tif'  

# DARK_IMG='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/dark_1s_002.tif'
# REF_IMG='/home/grizolli/DATA/20180807_RealMirror/20180807/mirror_mono12keV_5mrad/timescan_turn_all_off_from_all_on/CB4p8pi_84mm_12keV_1s_every5s_000.tif'

REF_IMG='/home/grizolli/DATA/20181019_RealMirror/20181020/06_gap28p5_correction/CB4p8pi_84mm_0p4apert_Baseline_opengap_009.tif'

for f in $FOLDER$FPATTERN
do

    SAMPLE_IMG=$f/*.tif
    
    #     DIST=$(echo $SAMPLE_IMG | cut -d '_' -f 14 | cut -d 'p' -f 1)

    echo 'Sample'
    echo $SAMPLE_IMG
    echo 'Reference:'
    echo $REF_IMG
    echo 'Dark:'
    echo $DARK_IMG
    
    parallel --bg -j 7 $SCRIPT $SAMPLE_IMG $REF_IMG $DARK_IMG 0.31 4.8 Edge 84.0 12.0 100.0 1 0 0 0 0 0 1
	
    paplay /usr/share/sounds/freedesktop/stereo/message.oga 

    sleep 3  # avoid some racing when accessing ini file
    
#     REF_IMG=$SAMPLE_IMG   
    
done 

parallel  --wait; paplay /usr/share/sounds/freedesktop/stereo/complete.oga; zenity --warning --text='\tDone!\t\t'



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

