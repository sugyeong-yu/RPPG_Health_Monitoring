import cv2
import heartpy as hp
import numpy as np
import scipy
from scipy import interpolate
from scipy import signal
from scipy import signal
from tensorflow.keras.models import load_model

# 주파수 필터링 함수 ##################################################
def temporal_bandpass_filter(data, fs, minBPM, maxBPM):
    try:
        nyq = 60 * fs / 2
        coef_vector = signal.butter(5, [minBPM / nyq, maxBPM / nyq], 'bandpass')
        return signal.filtfilt(*coef_vector, data)
    except ValueError:
        return data


# chrom 추출 #########################################################
norm = cv2.filter2D(np.ones((150, 3), np.float64), -1, np.ones((150, 30)))
def detrending(data, wsize):
    try:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        # n_channel = data.shape[0]
        # norm = signal.convolve2d(np.ones_like(data), np.ones((n_channel, wsize)), mode='same')
        # norm = cv2.filter2D(np.ones_like(data), -1, np.ones((n_channel, wsize)))

        # mean = signal.convolve2d(data, np.ones((n_channel, wsize)), mode='same') / norm
        mean = cv2.filter2D(data, -1, np.ones((150, wsize))) / norm

        return (data - mean) / (mean + 1e-15)
    except ValueError:
        return data


def CHROM(rgb_signal, fs, minBPM, maxBPM):
    # CHROM 검출
    raw_signal = np.array(rgb_signal).transpose()
    detrended = detrending(raw_signal, fs)
    detrended = detrended.transpose()
    # detrended = np.array(rgb_signal)
    X = 3 * detrended[0] - 2 * detrended[1]
    Y = 1.5 * detrended[0] + detrended[1] - 1.5 * detrended[2]
    Xf = temporal_bandpass_filter(X, fs, minBPM, maxBPM)
    Yf = temporal_bandpass_filter(Y, fs, minBPM, maxBPM)

    alpha = np.std(Xf) / np.std(Yf)
    chrom_signal = Xf - alpha * Yf

    return chrom_signal


# rppg 복원에 사용되는 함수들##############################################

def detect_peak(hrdata, distance, threshold):
    # point[0]가 peak의 x좌표 hrdata[point[0]]이 peak의 y좌표
    point = scipy.signal.find_peaks(hrdata, distance=distance, threshold=threshold)

    # peak = peak 값 (y축값), point[0] = peak의 x위치
    peak = np.zeros(len(point[0]))
    for i in range(len(point[0])):
        peak[i] = (hrdata[point[0][i]])
    return (peak, point[0])


def single_cycle_unit_rppg(rppg):
    rppg_data = []
    rppg_inverse = np.dot(rppg, -1)
    rpeaks, rpoints = detect_peak(np.array(rppg_inverse), 15, (0, 29))

    for i in range(len(rpoints) - 1):
        rppg_data.append(rppg[rpoints[i]:rpoints[i + 1]])

    return rppg_data


def feature(ppg):
    ppg_data = []
    for i in range(len(ppg)):
        data = []
        ppg[i] = ((ppg[i] - min(ppg[i])) / (max(ppg[i]) - min(ppg[i])))
        point = np.linspace(0, len(ppg[i]) - 1, 30)
        x = np.linspace(0, len(ppg[i]) - 1, len(ppg[i]))
        f = interpolate.interp1d(x, ppg[i], 'cubic')
        for j in range(30):
            data.append(f(point[j]))
        ppg_data.append(data)
    return ppg_data


def model_load(path):
    return load_model(path)

# hrv에 사용되는 함수들######################################################

# peak 찾는 함수(amplitude)
def detect_peak_hrv(ppg_time,ppg_data,distance):
    point = scipy.signal.find_peaks(ppg_data, distance=distance)
    peak = np.zeros(len(point[0]))
    peak_x = np.zeros(len(point[0]))
    for i in range(len(point[0])):
        peak_x[i] = ppg_time[point[0][i]]
        peak[i] = (ppg_data[point[0][i]])
# peak = peak 값 (y축값), point[0] = peak의 x위치
    return peak, peak_x

def preprocessing(ppg_data, cut_l, cut_h, sr):
    filtered = hp.filter_signal(ppg_data, cutoff=cut_l, sample_rate=sr, order=3, filtertype='lowpass')
    filtered = hp.filter_signal(filtered, cutoff=cut_h, sample_rate=sr, order=3, filtertype='highpass')
    return filtered

