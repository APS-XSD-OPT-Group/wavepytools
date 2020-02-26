#! /bin/bash

SCRIPT='/c/Users/zqiao/Documents/GitHub/wavepytools/wavepytools/imaging/single_grating/SingleGrating.py'
FOLDER='/c/Users/zqiao/Documents/xray_wavefront/20190717_ZoomOptics/20190721/Jtec_resp_dv200_P9/images/cropped/'
FOLDER_DARK='/c/Users/zqiao/Documents/xray_wavefront/20190717_ZoomOptics/20190721/'
FPATTERN='*.tif'  
# file pattern

DARK_IMG='dark_2000x500_8s.tif'
REF_IMG=${LISTOFFILES[0]}

for f in $FOLDER$FPATTERN
do

    SAMPLE_IMG=$f

    echo 'Sample'
    echo $SAMPLE_IMG
    echo 'Reference:'
    echo $REF_IMG
    echo 'Dark:'
    echo $DARK_IMG
    
    $SCRIPT $SAMPLE_IMG $REF_IMG $DARK_IMG 0.65 4.8 Diag 70.0 14.0 -0.7 1 0 0 0 0 0 0
    
    REF_IMG=$SAMPLE_IMG   
    
done

# FOLDER='/c/Users/zqiao/Documents/xray_wavefront/20190717_ZoomOptics/20190721/Jtec_resp_dv200_P6/images/cropped/'
FOLDER_RESULT='/c/Users/zqiao/Documents/xray_wavefront/20190717_ZoomOptics/20190721/Jtec_resp_dv200_P9/RF/'
cd $FOLDER
mkdir -p $FOLDER_RESULT'csv_WF' $FOLDER_RESULT'pngs'
cp 'resp_001_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch01.csv'
cp 'resp_003_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch02.csv'
cp 'resp_005_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch03.csv'
cp 'resp_007_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch04.csv'
cp 'resp_009_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch05.csv'
cp 'resp_011_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch06.csv'
cp 'resp_013_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch07.csv'
cp 'resp_015_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch08.csv'
cp 'resp_017_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch09.csv'
cp 'resp_019_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch10.csv'
cp 'resp_021_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch11.csv'
cp 'resp_023_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch12.csv'
cp 'resp_025_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch13.csv'
cp 'resp_027_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch14.csv'
cp 'resp_029_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch15.csv'
cp 'resp_031_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch16.csv'
cp 'resp_033_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch17.csv'
cp 'resp_035_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_integrated_y_01.csv' $FOLDER_RESULT'csv_WF/ch18.csv'

cp 'resp_001_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch01.png'
cp 'resp_003_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch02.png'
cp 'resp_005_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch03.png'
cp 'resp_007_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch04.png'
cp 'resp_009_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch05.png'
cp 'resp_011_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch06.png'
cp 'resp_013_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch07.png'
cp 'resp_015_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch08.png'
cp 'resp_017_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch09.png'
cp 'resp_019_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch10.png'
cp 'resp_021_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch11.png'
cp 'resp_023_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch12.png'
cp 'resp_025_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch13.png'
cp 'resp_027_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch14.png'
cp 'resp_029_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch15.png'
cp 'resp_031_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch16.png'
cp 'resp_033_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch17.png'
cp 'resp_035_output/profiles/TalbotImaging_cb3p39um_halfPi_d70mm_14p0KeV_dpc_profiles_Y_03.png' $FOLDER_RESULT'pngs/ch18.png'


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

