
############################ LIBRARIES ############################

import torch
import numpy as np
from random import randint, random

############################ PARAMETERS ############################


image_width = 150
image_height = 110
SAMPLE_RATE = 16000
NUM_MEL_BINS = 64
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01 
# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


############################# FUNCTIONS ############################

def hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(warper=1, num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))

    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    
    for spec_bin in spectrogram_bins_hertz:

        if spec_bin <= upper_edge_hertz:

            spec_bin *= warper
        
        else:

            spec_bin = nyquist_hertz - (nyquist_hertz-upper_edge_hertz*min(warper,1))*(nyquist_hertz-spec_bin)/(nyquist_hertz-upper_edge_hertz*min(warper,1)/warper)

    
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)

    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))

    for i in range(num_mel_bins):

        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
    mel_weights_matrix[0, :] = 0.0

    return mel_weights_matrix.astype(np.float32)


def augment_spectrogram(spectrogram, warper):

    mel_spectrogram = torch.mm(spectrogram, torch.from_numpy(spectrogram_to_mel_matrix(
        warper=warper,
        num_spectrogram_bins=np.shape(spectrogram)[1],
        audio_sample_rate=SAMPLE_RATE,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ)) )

    return torch.log(mel_spectrogram + LOG_OFFSET)

