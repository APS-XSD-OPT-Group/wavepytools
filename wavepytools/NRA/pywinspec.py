''' winspec.py - read SPE files created by WinSpec with Princeton Instruments' cameras.
https://github.com/antonl/pyWinSpec
'''


import ctypes, os
import numpy as np

__all__ = ['SpeFile', 'test_headers']

__author__ = "Anton Loukianov"
__email__ = "anton.loukianov@gmail.com"
__license__ = "BSD"
__version__ = "0.1"

# Definitions of types
spe_byte = ctypes.c_ubyte
spe_word = ctypes.c_ushort
spe_dword = ctypes.c_uint

spe_char = ctypes.c_char # 1 byte
spe_short = ctypes.c_short # 2 bytes

# long is 4 bytes in the manual. It is 8 bytes on my machine
spe_long = ctypes.c_int # 4 bytes

spe_float = ctypes.c_float # 4 bytes
spe_double = ctypes.c_double # 8 bytes

class ROIinfo(ctypes.Structure):
    pass

class AxisCalibration(ctypes.Structure):
    pass

class Header(ctypes.Structure):
    pass

def test_headers():
    ''' Print the attribute names, sizes and offsets in the C structure
    
    Assuming that the sizes are correct and add up to an offset of 4100 bytes, 
    everything should add up correctly. This information was taken from the 
    WinSpec 2.6 Spectroscopy Software User Manual version 2.6B, page 251.

    If this table doesn't add up, something changed in the definitions of the 
    datatype widths. Fix this in winspec.structs file and let me know!
    '''

    import inspect, re

    A = Header()
    
    for i in [Header, AxisCalibration, ROIinfo]:
        fields = []

        print('\n{:30s}[{:4s}]\tsize'.format(i, 'offs'))
        
        for name,obj in inspect.getmembers(i):
            if inspect.isdatadescriptor(obj) and not inspect.ismemberdescriptor(obj) \
                and not inspect.isgetsetdescriptor(obj):
                
                fields.append((name, obj))

        fields.sort(key=lambda x: re.search('(?<=ofs=)([0-9]+)', str(x[1])).group(0), 
                cmp=lambda x,y: cmp(int(x),int(y))); fields

        for name, obj in fields:
            print('{:30s}[{:4d}]\t{:4d}'.format(name, obj.size, obj.offset))

class SpeFile(object):
    ''' A file that represents the SPE file.

    All details written in the file are contained in the `header` structure. Data is 
    accessed by using the `data` property.

    Once the object is created and data accessed, the file is NOT read again. Create
    a new object if you want to reread the file.
    '''

    # Map between header datatype field and numpy datatype 
    _datatype_map = {0 : np.float32, 1 : np.int32, 2 : np.int16, 3 : np.uint16}

    def __init__(self, name):
        ''' Open file `name` to read the header.'''

        with open(name, mode='rb') as f:
            self.header = Header()
            self.path = os.path.realpath(name) 
            self._data = None

            # Deprecated method, but FileIO apparently can't be used with numpy
            f.readinto(self.header)

    def _read(self):
        ''' Read the data segment of the file and create an appropriately-shaped numpy array

        Based on the header, the right datatype is selected and returned as a numpy array.  I took 
        the convention that the frame index is the first, followed by the x,y coordinates.
        '''

        if self._data is not None:
            return self._data

        # In python 2.7, apparently file and FileIO cannot be used interchangably
        with open(self.path, mode='rb') as f:
            f.seek(4100) # Skip header (4100 bytes)

            _count = self.header.xdim * self.header.ydim * self.header.NumFrames
            
            self._data = np.fromfile(f, dtype=SpeFile._datatype_map[self.header.datatype], count=_count)

            # Also, apparently the ordering of the data corresponds to how it is stored by the shift register
            # Thus, it appears a little backwards...
            self._data = self._data.reshape((self.header.NumFrames, self.header.ydim, self.header.xdim))

            # Orient the structure so that it is indexed like [NumFrames][x, y]
            self._data = np.rollaxis(self._data, 2, 1)

            return self._data

    ''' Data recorded in the file, returned as a numpy array. 
    
    The convention for indexes is that the first index is the frame index, followed by x,y region of 
    interest.
    '''
    data = property(fget=_read)

    def __repr__(self):
        return 'SPE File {:s}\n\t{:d}x{:d} area, {:d} frames\n\tTaken on {:s}' \
                .format(self.path, self.header.xdim, self.header.ydim, self.header.NumFrames, self.header.date)

# Lengths of arrays used in header
HDRNAMEMAX = 120
USERINFOMAX = 1000
COMMENTMAX = 80
LABELMAX = 16
FILEVERMAX = 16
DATEMAX = 10
ROIMAX = 10
TIMEMAX = 7

# Definitions of WinSpec structures

# Region of interest defs
ROIinfo._pack_ = 1
ROIinfo._fields_ = [
    ('startx', spe_word), 
    ('endx', spe_word),
    ('groupx', spe_word),
    ('starty', spe_word),
    ('endy', spe_word),
    ('groupy', spe_word)]

# Calibration structure for X and Y axes
AxisCalibration._pack_ = 1
AxisCalibration._fields_ = [
    ('offset', spe_double), 
    ('factor', spe_double),
    ('current_unit', spe_char),
    ('reserved1', spe_char),
    ('string', spe_char * 40),
    ('reserved2', spe_char * 40), 
    ('calib_valid', spe_char),
    ('input_unit', spe_char),
    ('polynom_unit', spe_char),
    ('polynom_order', spe_char),
    ('calib_count', spe_char),
    ('pixel_position', spe_double * 10),
    ('calib_value', spe_double * 10), 
    ('polynom_coeff', spe_double * 6),
    ('laser_position', spe_double),
    ('reserved3', spe_char),
    ('new_calib_flag', spe_byte),
    ('calib_label', spe_char * 81),
    ('expansion', spe_char * 87)]

# Full header definition
Header._pack_ = 1
Header._fields_ = [
    ('ControllerVersion', spe_short),
    ('LogicOutput', spe_short),
    ('AmpHiCapLowNoise', spe_word),
    ('xDimDet', spe_word),
    ('mode', spe_short),
    ('exp_sec', spe_float),
    ('VChipXdim', spe_short),
    ('VChipYdim', spe_short),
    ('yDimDet', spe_word),
    ('date', spe_char * DATEMAX),
    ('VirtualChipFlag', spe_short),
    ('Spare_1', spe_char * 2), # Unused data
    ('noscan', spe_short),
    ('DetTemperature', spe_float),
    ('DetType', spe_short),
    ('xdim', spe_word),
    ('stdiode', spe_short),
    ('DelayTime', spe_float),
    ('ShutterControl', spe_word),
    ('AbsorbLive', spe_short),
    ('AbsorbMode', spe_word),
    ('CanDoVirtualChipFlag', spe_short),
    ('ThresholdMinLive', spe_short),
    ('ThresholdMinVal', spe_float), 
    ('ThresholdMaxLive', spe_short), 
    ('ThresholdMaxVal', spe_float),
    ('SpecAutoSpectroMode', spe_short),
    ('SpecCenterWlNm', spe_float),
    ('SpecGlueFlag', spe_short),
    ('SpecGlueStartWlNm', spe_float),
    ('SpecGlueEndWlNm', spe_float),
    ('SpecGlueMinOvrlpNm', spe_float),
    ('SpecGlueFinalResNm', spe_float),
    ('PulserType', spe_short),
    ('CustomChipFlag', spe_short),
    ('XPrePixels', spe_short),
    ('XPostPixels', spe_short),
    ('YPrePixels', spe_short),
    ('YPostPixels', spe_short),
    ('asynen', spe_short),
    ('datatype', spe_short), # 0 - float, 1 - long, 2 - short, 3 - ushort
    ('PulserMode', spe_short),
    ('PulserOnChipAccums', spe_word),
    ('PulserRepeatExp', spe_dword),
    ('PulseRepWidth', spe_float),
    ('PulseRepDelay', spe_float),
    ('PulseSeqStartWidth', spe_float),
    ('PulseSeqEndWidth', spe_float),
    ('PulseSeqStartDelay', spe_float),
    ('PulseSeqEndDelay', spe_float),
    ('PulseSeqIncMode', spe_short),
    ('PImaxUsed', spe_short),
    ('PImaxMode', spe_short),
    ('PImaxGain', spe_short),
    ('BackGrndApplied', spe_short),
    ('PImax2nsBrdUsed', spe_short),
    ('minblk', spe_word),
    ('numminblk', spe_word),
    ('SpecMirrorLocation', spe_short * 2),
    ('SpecSlitLocation', spe_short * 4),
    ('CustomTimingFlag', spe_short),
    ('ExperimentTimeLocal', spe_char * TIMEMAX),
    ('ExperimentTimeUTC', spe_char * TIMEMAX),
    ('ExposUnits', spe_short),
    ('ADCoffset', spe_word),
    ('ADCrate', spe_word),
    ('ADCtype', spe_word),
    ('ADCresolution', spe_word),
    ('ADCbitAdjust', spe_word),
    ('gain', spe_word),
    ('Comments', spe_char * 5 * COMMENTMAX),
    ('geometric', spe_word), # x01 - rotate, x02 - reverse, x04 flip
    ('xlabel', spe_char * LABELMAX),
    ('cleans', spe_word),
    ('NumSkpPerCln', spe_word),
    ('SpecMirrorPos', spe_short * 2),
    ('SpecSlitPos', spe_float * 4), 
    ('AutoCleansActive', spe_short),
    ('UseContCleansInst', spe_short),
    ('AbsorbStripNum', spe_short), 
    ('SpecSlipPosUnits', spe_short),
    ('SpecGrooves', spe_float),
    ('srccmp', spe_short),
    ('ydim', spe_word), 
    ('scramble', spe_short),
    ('ContinuousCleansFlag', spe_short), 
    ('ExternalTriggerFlag', spe_short), 
    ('lnoscan', spe_long), # Longs are 4 bytes  
    ('lavgexp', spe_long), # 4 bytes
    ('ReadoutTime', spe_float), 
    ('TriggeredModeFlag', spe_short), 
    ('Spare_2', spe_char * 10), 
    ('sw_version', spe_char * FILEVERMAX), 
    ('type', spe_short),
    ('flatFieldApplied', spe_short), 
    ('Spare_3', spe_char * 16), 
    ('kin_trig_mode', spe_short), 
    ('dlabel', spe_char * LABELMAX), 
    ('Spare_4', spe_char * 436), 
    ('PulseFileName', spe_char * HDRNAMEMAX), 
    ('AbsorbFileName', spe_char * HDRNAMEMAX),
    ('NumExpRepeats', spe_dword),
    ('NumExpAccums', spe_dword),
    ('YT_Flag', spe_short), 
    ('clkspd_us', spe_float),
    ('HWaccumFlag', spe_short),
    ('StoreSync', spe_short),
    ('BlemishApplied', spe_short),
    ('CosmicApplied', spe_short), 
    ('CosmicType', spe_short),
    ('CosmicThreshold', spe_float), 
    ('NumFrames', spe_long),
    ('MaxIntensity', spe_float),
    ('MinIntensity', spe_float),
    ('ylabel', spe_char * LABELMAX),
    ('ShutterType', spe_word),
    ('shutterComp', spe_float),
    ('readoutMode', spe_word),
    ('WindowSize', spe_word),
    ('clkspd', spe_word),
    ('interface_type', spe_word),
    ('NumROIsInExperiment', spe_short),
    ('Spare_5', spe_char * 16),
    ('controllerNum', spe_word),
    ('SWmade', spe_word),
    ('NumROI', spe_short),
    ('ROIinfblk', ROIinfo * ROIMAX),
    ('FlatField', spe_char * HDRNAMEMAX),
    ('background', spe_char * HDRNAMEMAX),
    ('blemish', spe_char * HDRNAMEMAX),
    ('file_header_ver', spe_float),
    ('YT_Info', spe_char * 1000),
    ('WinView_id', spe_long),
    ('xcalibration', AxisCalibration),
    ('ycalibration', AxisCalibration),
    ('Istring', spe_char * 40),
    ('Spare_6', spe_char * 25),
    ('SpecType', spe_byte),
    ('SpecModel', spe_byte),
    ('PulseBurstUsed', spe_byte),
    ('PulseBurstCount', spe_dword),
    ('PulseBurstPeriod', spe_double),
    ('PulseBracketUsed', spe_byte),
    ('PulseBracketType', spe_byte),
    ('PulseTimeConstFast', spe_double),
    ('PulseAmplitudeFast', spe_double),
    ('PulseTimeConstSlow', spe_double),
    ('PulseAmplitudeSlow', spe_double),
    ('AnalogGain', spe_short),
    ('AvGainUsed', spe_short),
    ('AvGain', spe_short),
    ('lastvalue', spe_short)]

