'''
Deals with the actual detection of signals in multichannel audio files. 
There are two problems that need to solved while detecting a signal of interest.
    #. within-channel signal detection
    #. across-channel correspondence matching

Within-channel signal detection
-------------------------------
This task involves `locally` checking if there are any signals of interest in one channel at a time. The exact methods used for 
the within-channel can be set by the user, though the simplest is of course a basic threshold-type detector. Whenever the 
signal goes beyond a particular threshold, a signal is considered to be in that region.


Built-in detection routines
---------------------------
The detection module has a few simple detection routines. More advanced routines
are unlikely to form a core part of the package, and need to be written by the 
user. 

#. dBrms_detector : Calculates the moving dB rms profile of an audio clip. The
User needs to define the size of the moving window and the threshold in dB rms. 

#. envelope_detector : Generates the Hilbert envelop of the audio clip. Regions above
the set threshold in dB peak amplitude are defined as detections. This method is faster
than the dBrms_detector.
'''

import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']=10000
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import scipy.ndimage as ndimage
import tqdm

from batracker.common_dsp.sigproc import *


def cross_channel_threshold_detector(multichannel, fs, **kwargs):
    '''
    Parameters
    ----------
    multichannel : np.array
        Msamples x Nchannels audio data
    fs : float >0
    detector_function : function, optional 
        The function used to detect the start and end of a signal. 
        Any custom detector function can be given, the compulsory inputs
        are audio np.array, sample rate and the function should accept keyword
        arguments (even if it doesn't use them.)
        Defaults to dBrms_detector. 
    
    
    Returns
    -------
    all_detections : list
        A list with sublists containing start-stop times of the detections 
        in each channel. Each sublist contains the detections in one channel.
        
    Notes
    -----
    For further keyword arguments see the `threshold_detector` function
    
    See Also
    --------
    dBrms_detector
    
    '''
    samples, channels = multichannel.shape
    detector_function = kwargs.get('detector_function', dBrms_detector)
    print(channels, samples)
    all_detections = []
    for each in tqdm.tqdm(range(channels)):
        all_detections.append(detector_function(multichannel[:,each], fs, **kwargs))
    return all_detections
        



def dBrms_detector(one_channel, fs, **kwargs):
    '''
    Calculates the dB rms profile of the input audio and 
    selects regions which arae above  the profile. 
    
    Parameters
    ----------
    one_channel
    fs
    dbrms_threshold: float, optional
        Defaults to  -50 dB rms
    dbrms_window: float, optional
        The window which is used to calculate the dB rms profile
        in seconds.  Defaults to 0.001 seconds.
    
    Returns
    -------
    detections : list with tuples
        Each tuple corresponds to a candidate signal region
    '''
    if one_channel.ndim > 1:
        raise IndexError(f'Input audio must be flattened, and have only 1 dimension. \
                         Current audio has {one_channel.ndim} dimensions')
    dbrms_window = kwargs.get('dbrms_window',0.001) # seconds
    dbrms_threshold = kwargs.get('dbrms_threshold', -50)
    
    window_samples = int(fs*dbrms_window)
    dBrms_profile = dB(moving_rms(one_channel, window_size=window_samples))
    
    labelled, num_regions = ndimage.label(dBrms_profile>dbrms_threshold)
    if num_regions==0:
        print (f'No regions above threshold: {dbrms_threshold} dBrms found in this channel!')
    regions_above = ndimage.find_objects(labelled.flatten())
    regions_above_timestamps = [get_start_stop_times(each, fs) for each in regions_above]
    
    return regions_above_timestamps


def envelope_detector(audio, fs, **kwargs):
    '''
    Generates the Hilbert envelope of the audio. Signals are detected
    wherever the envelope goes beyond a user-defined threshold value.
    
    Two main options are to segment loud signals with reference to dB peak or 
    with reference dB above floor level. 
    
    Parameters
    ----------
    audio
    fs
    
    
    Keyword Arguments
    -----------------
    threshold_db_floor: float, optional
        The threshold for signal detection in dB above the floor level. The 5%ile level of the whole envelope is chosen as
        the floor level. If not specified, then threshold_dbpeak is used to segment signals.
    threshold_dbpeak : float, optional
        The value beyond which a signal is considered to start.
        Used only if relative_to_baseline is True.
    lowpass_durn: float, optional
        The highest time-resolution of envelope fluctuation to keep. 
        This effectively performs a low-pass at 1/lowpass_durn Hz on the raw envelope
        signal. 
    

    Returns
    -------
    regions_above_timestamps 
    
    
    
    '''
    envelope = np.abs(signal.hilbert(audio))
    
    
    if not kwargs.get('lowpass_durn') is None:
        lowpass_durn = kwargs['lowpass_durn'] # seconds
        freq = 1.0/lowpass_durn
        b,a = signal.butter(1, freq/(fs*0.5),'lowpass')
        envelope = signal.filtfilt(b,a,envelope)
    
    if not kwargs.get('threshold_db_floor', None) is None:
        floor_level = np.percentile(20*np.log10(envelope),5)
        threshold_db = floor_level + kwargs['threshold_db_floor']
    else:
        # get regions above the threshold
        threshold_db = kwargs['threshold_dbpeak']
    linear_threshold = 10**(threshold_db/20)
    labelled, num_detections = ndimage.label(envelope>=linear_threshold)
    regions_above = ndimage.find_objects(labelled.flatten())
    regions_above_timestamps = [get_start_stop_times(each, fs   ) for each in regions_above]
    return regions_above_timestamps


 
def get_start_stop_times(findobjects_tuple, fs):
    '''
    
    '''
    only_tuple = findobjects_tuple[0]
    start, stop = only_tuple.start/fs, only_tuple.stop/fs
    return start, stop


def moving_rms(X, **kwargs):
    '''Calculates moving rms of a signal with given window size. 
    Outputs np.array of *same* size as X. The rms of the 
    last few samples <= window_size away from the end are assigned
    to last full-window rms calculated
    Parameters
    ----------
    X :  np.array
        Signal of interest. 
    window_size : int, optional
                 Defaults to 125 samples. 
    Returns
    -------
    all_rms : np.array
        Moving rms of the signal. 
    '''
    window_size = kwargs.get('window_size', 125)
    starts = np.arange(0, X.size)
    stops = starts+window_size
    valid = stops<X.size
    valid_starts = np.int32(starts[valid])
    valid_stops = np.int32(stops[valid])
    all_rms = np.ones(X.size).reshape(-1,1)*999

    for i, (start, stop) in enumerate(zip(valid_starts, valid_stops)):
        rms_value = rms(X[start:stop])
        all_rms[i] = rms_value
    
    # replace all un-assigned samples with the last rms value
    all_rms[all_rms==999] = np.nan

    return all_rms
#    
#if __name__ == '__main__':
#    import scipy.signal as signal 
#    # trying out  the hilbert envelope method:
#    fs = 250000
#    background = -60 # dB rms
#    audio = np.random.normal(0, 10**(background/20), fs)
#    duration = 0.005
#    sound_start = 0.05
#    t = np.linspace(0, duration, int(fs*duration))
#    bat_call = signal.chirp(t,90000, 25000, t[-1])
#    bat_call *= 0.5
#    sound_stop = sound_start+duration
#    
#    start, end = np.int32(np.array([sound_start,
#                                    sound_stop])*fs)
#    audio[start:end] += bat_call
#    
#    envelope = np.abs(signal.hilbert(audio))
#    
#    dets = envelope_detector(audio, fs, threshold_dbpeak=-20)
#    print(dets)
##        