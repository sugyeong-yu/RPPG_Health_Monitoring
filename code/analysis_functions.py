#신호에서 추출할 수 있는 생체 정보 구하는 함수들
#평균 맥박수, rppg, spo2, hrv 구하는 함수

import numpy as np
from hrvanalysis import get_time_domain_features
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import functions

#평균 맥박수 함수
def estimate_average_pulserate(data, fs, minBPM, maxBPM):
    # 평균 맥박수 측정
    f, pxx = signal.periodogram(data, fs=fs, window='hann')
    max_peak_idx = np.argmax(pxx)
    bpm = int(f[max_peak_idx] * 60)
    return min(max(bpm, minBPM), maxBPM)

# rppg 신호 복원 함수
def ppg_restore(rppg, plot_height,model_svr,model_sd,restore_rppg_buffer):
    restore_rppg = []

    rppg_cycle = functions.single_cycle_unit_rppg(rppg)
    rppg30 = functions.feature(rppg_cycle)

    train_rppg_svr = model_svr.predict(np.reshape(rppg30, (-1, 30)))
    train_rppg_sd = model_sd.predict(np.reshape(train_rppg_svr, (-1, 30)))

    for i in train_rppg_sd:
        restore_rppg.extend(i)

    minmax = ((restore_rppg - min(restore_rppg)) / (max(restore_rppg) - min(restore_rppg))) * plot_height

    restore_rppg_buffer.extend(minmax)

#spo2 추출 함수
def rspo2_extract(rgb_signal, fs, minBPM, maxBPM, rspo2_buffer):
    r_crop = np.array(rgb_signal[0]) / 255
    g_crop = np.array(rgb_signal[1]) / 255
    b_crop = np.array(rgb_signal[2]) / 255

    # y = ((65.481 * r_crop) + (128.553 * g_crop) + (24.966 * b_crop)) + 16
    cropped_cg = ((-81.085 * r_crop) + (112 * g_crop) + (-30.915 * b_crop)) + 128
    cropped_cr = ((112 * r_crop) + (-93.786 * g_crop) + (-18.214 * b_crop)) + 128

    cr_pass = functions.temporal_bandpass_filter(cropped_cr, fs, minBPM, maxBPM)
    cg_pass = functions.temporal_bandpass_filter(cropped_cg, fs, minBPM, maxBPM)

    cr_p = cr_pass
    cr_v = -cr_pass
    cg_p = cg_pass
    cg_v = -cg_pass

    cr_peaks, _ = find_peaks(cr_p, distance=15)
    cr_valleys, _ = find_peaks(cr_v, distance=15)
    cg_peaks, _ = find_peaks(cg_p, distance=15)
    cg_valleys, _ = find_peaks(cg_v, distance=15)

    cr_length = len(cr_peaks) if len(cr_peaks) < len(cr_valleys) else len(cr_valleys)
    cg_length = len(cg_peaks) if len(cg_peaks) < len(cg_valleys) else len(cg_valleys)

    cr_peak2valley = []
    cg_peak2valley = []

    for r in range(cr_length):
        ampl = np.abs(cr_p[cr_valleys[r]] / cr_p[cr_peaks[r]])
        cr_peak2valley.append(ampl)

    for g in range(cg_length):
        ampl = np.abs(cg_p[cg_valleys[g]] / cg_p[cg_peaks[g]])
        cg_peak2valley.append(ampl)

    ac_cr = np.median(cr_peak2valley)
    ac_cg = np.median(cg_peak2valley)

    cr_log = np.log(np.array(ac_cr) + 1)
    cg_log = np.log(np.array(ac_cg) + 1)

    ratio = cr_log / cg_log

    # rspo2 = int(11.8805 * ratio + 79.1915)
    # rspo2 = int(11.8805 * ratio + 85.1914)
    rspo2 = int(19.8805 * ratio + 72.9847)
    # rspo2 = int(50.60 * ratio - 11.81 * (ratio**2) + 47.19)


    rspo2_buffer.append(rspo2)

    plot_rspo2 = None
    if len(rspo2_buffer) >= 20:
        avg_spo2 = int(np.average(rspo2_buffer))
        if avg_spo2 >= 100:
            avg_spo2 = 99
        plot_rspo2 = avg_spo2
        if len(rspo2_buffer) >30:
            del rspo2_buffer[0]

    return plot_rspo2


#hrv 분석 함수
def hrv_analysis(signal, times, sr=30, distance=100):
    rppg_sig_ = np.array(signal, dtype='float32')
    rppg_time_ = np.array(times, dtype='float32')  # msec단위
    # ================= preprocessing===================
    # 1. interpolation
    rppg_time = np.linspace(rppg_time_[0], rppg_time_[-1], len(rppg_time_) * 8)  # interpolation된 x
    #
    i = interp1d(rppg_time_, rppg_sig_, kind='quadratic')
    rppg_sig = i(rppg_time)  # interpolation된 y

    sr=sr*8
    # 2. bandpass filtering
    filtered = functions.preprocessing(rppg_sig, 2.0, 0.5, sr)
    r_peaks_y, r_peaks_x = functions.detect_peak_hrv(rppg_time, filtered, distance)

    # 3. extract PPI
    rppg_ppi = np.diff(r_peaks_x)

    # ================ hrv analysis ======================
    hrv_feature = get_time_domain_features(rppg_ppi)  # rppg_ppi r_nni

    return hrv_feature
