from scipy.signal import butter, lfilter
import numpy as np

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=3.667, fs=30, order=6):
    first = data[0]
    data = np.array(data)
    data -= first
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    y += first
    return y


