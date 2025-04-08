import numpy as np
from scipy.signal import correlate



def compare_waveforms(signal1, signal2):
    # Normalize signals
    signal1 = signal1 / np.max(np.abs(signal1))
    signal2 = signal2 / np.max(np.abs(signal2))
    # Compute cross-correlation
    correlation = correlate(signal1, signal2, mode='full')
    max_corr = np.max(correlation)
    return max_corr

