import cv2
import numpy as np
import os
import json
from scipy import signal
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
import csv

import pickle
import others

from util import kcf
from util import skinsegment
from hrv.hrv_analysis import *


def interp_1d(arr_short,arr_long):
  return np.interp(
    np.arange(0,arr_long.shape[0]),
    np.linspace(0,arr_long.shape[0],num=arr_short.shape[0]),
    arr_short)

def detrending(data, wsize):
    try:
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        n_channel = data.shape[0]
        norm = signal.convolve2d(np.ones_like(data), np.ones((n_channel, wsize)), mode='same')
        mean = signal.convolve2d(data, np.ones((n_channel, wsize)), mode='same') / norm
        return (data - mean) / (mean + 1e-15)
    except ValueError:
        return data

def temporal_bandpass_filter(data, fs, minBPM, maxBPM):
    try:
        nyq = 60 * fs / 2
        coef_vector = signal.butter(5, [minBPM / nyq, maxBPM / nyq], 'bandpass')
        return signal.filtfilt(*coef_vector, data)
    except ValueError:
        return data

def estimate_average_pulserate(data, fs, minBPM, maxBPM):
    # 평균 맥박수 측정
    f, pxx = signal.periodogram(data, fs=fs, window='hann')
    max_peak_idx = np.argmax(pxx)
    bpm = int(f[max_peak_idx] * 60)
    return min(max(bpm, minBPM), maxBPM)

def calculate_snr(data, fs, fundamental_freq, use_harmonic=False):
    f, pxx = signal.periodogram(data, fs=fs, window='hann')

    fundamental_range = (fundamental_freq - 2, fundamental_freq + 2)
    energy_of_interest = np.sum(pxx[fundamental_range[0]:fundamental_range[1]])

    if use_harmonic:
        harmonic_freq = 2 * fundamental_freq
        harmonic_range = (harmonic_freq - 5, harmonic_freq + 5)
        energy_of_interest += np.sum(pxx[harmonic_range[0]:harmonic_range[1]])

    energy_of_remaining = np.sum(pxx) - energy_of_interest
    ratio = energy_of_interest / (energy_of_remaining + 1e-17)
    snr = 10 * np.log10(ratio)

    return snr

def CHROM(rgb_signal, fs, minBPM, maxBPM):
    # CHROM 검출
    raw_signal = np.array(rgb_signal).transpose()
    detrended = detrending(raw_signal, fs)
    detrended = detrended.transpose()
    X = 3 * detrended[0] - 2 * detrended[1]
    Y = 1.5 * detrended[0] + detrended[1] - 1.5 * detrended[2]
    Xf = temporal_bandpass_filter(X, fs, minBPM, maxBPM)
    Yf = temporal_bandpass_filter(Y, fs, minBPM, maxBPM)
    alpha = np.std(Xf) / np.std(Yf)
    chrom_signal = Xf - alpha * Yf
    # 주피수 스펙트럼
    f, chrom_spectrum = signal.periodogram(chrom_signal, fs=fs, window='hann')


    return f, chrom_spectrum, chrom_signal



def run(file_path):
    is_tracking = False
    track_toler = 1
    prev_bbox = [0, 0, 0, 0]
    detect_th = 0.5
    detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
    tracker = kcf.KCFTracker()

    r_buffer = []
    g_buffer = []
    b_buffer = []

    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(file_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        _, frame = cap.read()
        if _ == False:
            break

        curr_bbox = []
        # 검출된 얼굴이 존재하지 않는 경우 검출 수행
        if not is_tracking:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False,
                                         True)
            detector.setInput(blob)
            detections = detector.forward()
            bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                      detections[0, 0, i, 2] >= detect_th]
            if len(bboxes) > 0:
                # 가장 큰 얼굴 하나만 검출
                bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
                curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                             bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 트래커에 현재 얼굴 위치 등록
                tracker.init(frame, curr_bbox)
                is_tracking = True

        # 검출된 얼굴이 존재하는 경우 위치 업데이트
        elif is_tracking:
            is_tracking, curr_bbox = tracker.update(frame)
            curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in zip(curr_bbox, prev_bbox)]
            prev_bbox = curr_bbox

        try:
            face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            mask = skinsegment.create_skin_mask(face)
            n_pixel = max(1, np.sum(mask))

            b, g, r = cv2.split(face)
            r[mask == 0] = 0
            g[mask == 0] = 0
            b[mask == 0] = 0

            r = r.astype(np.float32)
            g = g.astype(np.float32)
            b = b.astype(np.float32)

            r_mean = np.sum(r) / n_pixel
            g_mean = np.sum(g) / n_pixel
            b_mean = np.sum(b) / n_pixel

            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)

            cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                          (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 2)

        except:
            r_buffer.append(0.0)
            g_buffer.append(0.0)
            b_buffer.append(0.0)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

    return [r_buffer, g_buffer, b_buffer]


def real_time_plot(file_path, fs, minBPM, maxBPM):
    is_tracking = False
    track_toler = 1
    prev_bbox = [0, 0, 0, 0]
    detect_th = 0.5
    detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
    tracker = kcf.KCFTracker()

    r_buffer = []
    g_buffer = []
    b_buffer = []

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(file_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ##############

    stack = 0
    data_size = 100

    x_plot = np.linspace(0, data_size-1, data_size)
    y_plot = np.zeros(data_size)

    plt.ion()  # 대화식 모드를 켠다.

    figure, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x_plot, y_plot)

    plt.xlim([0, 50])
    plt.ylim([-0.01, 0.01])

    past_time = 0

    ##############

    while True:
        stack += 1

        _, frame = cap.read()
        now_time = time.time()
        if past_time != 0:
            time_inverse = now_time-past_time
            fs = int(1/time_inverse)
            print(fs)
        past_time = now_time

        if _ == False:
            break

        curr_bbox = []
        # 검출된 얼굴이 존재하지 않는 경우 검출 수행
        if not is_tracking:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False,
                                         True)
            detector.setInput(blob)
            detections = detector.forward()
            bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                      detections[0, 0, i, 2] >= detect_th]
            if len(bboxes) > 0:
                # 가장 큰 얼굴 하나만 검출
                bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
                curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                             bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 트래커에 현재 얼굴 위치 등록
                tracker.init(frame, curr_bbox)
                is_tracking = True

        # 검출된 얼굴이 존재하는 경우 위치 업데이트
        elif is_tracking:
            is_tracking, curr_bbox = tracker.update(frame)
            curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in zip(curr_bbox, prev_bbox)]
            prev_bbox = curr_bbox

        try:
            face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            mask = skinsegment.create_skin_mask(face)
            n_pixel = max(1, np.sum(mask))

            b, g, r = cv2.split(face)
            r[mask == 0] = 0
            g[mask == 0] = 0
            b[mask == 0] = 0

            r = r.astype(np.float32)
            g = g.astype(np.float32)
            b = b.astype(np.float32)

            r_mean = np.sum(r) / n_pixel
            g_mean = np.sum(g) / n_pixel
            b_mean = np.sum(b) / n_pixel

            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)

            cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                          (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 2)

        except:
            r_buffer.append(0.0)
            g_buffer.append(0.0)
            b_buffer.append(0.0)

        # 일정한 스택이 쌓이면 실행
        if stack >= data_size:
            # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
            rgb_signal = [r_buffer[-data_size:], g_buffer[-data_size:], b_buffer[-data_size:]]
            f, chrom_spectrum, rPPG = CHROM(rgb_signal, fs, minBPM, maxBPM)

            # y업데이트하고 그리기
            y_plot = rPPG
            line.set_xdata(x_plot)
            line.set_ydata(y_plot)
            figure.canvas.draw()
            figure.canvas.flush_events()

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 32:
            plt.close()
            break


    cv2.destroyAllWindows()


def real_time_cv2(file_path, fs, minBPM, maxBPM):
    is_tracking = False
    track_toler = 1
    prev_bbox = [0, 0, 0, 0]
    detect_th = 0.5
    detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
    tracker = kcf.KCFTracker()

    r_buffer = []
    g_buffer = []
    b_buffer = []

    # 실시간 or 파일 불러와서
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(file_path)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ##############

    stack = 0
    data_size = 150

    plot_width = int(w*0.7)
    plot_height = int(h*0.15)
    plot_x = int(w/2 - plot_width/2)
    plot_y = int(h*0.8)

    plot_color = (255, 0, 0)
    plot_val = []

    plot_scale = np.linspace(0, plot_width-1, plot_width)

    past_time = 0
    webcam = False # 프레임레이트 실시간 계산하는거 할거면 True

    ##############

    while True:
        stack += 1

        _, frame = cap.read()
        now_time = time.time()
        if webcam:
            if past_time != 0:
                time_inverse = now_time-past_time
                fs = int(1/time_inverse)
                print(fs)
            past_time = now_time

        if _ == False:
            break

        curr_bbox = []
        # 검출된 얼굴이 존재하지 않는 경우 검출 수행
        if not is_tracking:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False,
                                         True)
            detector.setInput(blob)
            detections = detector.forward()
            bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                      detections[0, 0, i, 2] >= detect_th]
            if len(bboxes) > 0:
                # 가장 큰 얼굴 하나만 검출
                bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
                curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                             bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 트래커에 현재 얼굴 위치 등록
                tracker.init(frame, curr_bbox)
                is_tracking = True

        # 검출된 얼굴이 존재하는 경우 위치 업데이트
        elif is_tracking:
            is_tracking, curr_bbox = tracker.update(frame)
            curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in zip(curr_bbox, prev_bbox)]
            prev_bbox = curr_bbox

        try:
            face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            mask = skinsegment.create_skin_mask(face)
            n_pixel = max(1, np.sum(mask))

            b, g, r = cv2.split(face)
            r[mask == 0] = 0
            g[mask == 0] = 0
            b[mask == 0] = 0

            r = r.astype(np.float32)
            g = g.astype(np.float32)
            b = b.astype(np.float32)

            r_mean = np.sum(r) / n_pixel
            g_mean = np.sum(g) / n_pixel
            b_mean = np.sum(b) / n_pixel

            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)

            cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                          (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 2)

        except:
            r_buffer.append(0.0)
            g_buffer.append(0.0)
            b_buffer.append(0.0)

        # 일정한 스택이 쌓이면 실행
        if stack >= data_size:
            # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
            rgb_signal = [r_buffer[-data_size:], g_buffer[-data_size:], b_buffer[-data_size:]]
            f, chrom_spectrum, rPPG = CHROM(rgb_signal, fs, minBPM, maxBPM)

            # y업데이트하고 그리기
            minmax = ((rPPG - min(rPPG)) / (max(rPPG) - min(rPPG)))*plot_height
            minmax = interp_1d(minmax, plot_scale) # 플롯된 길이만큼 늘리기

            plot_val.extend(minmax)

            while len(plot_val) > plot_width:
                del plot_val[0]

            cv2.rectangle(frame, (plot_x, plot_y, plot_width, plot_height), (255, 255, 255), -1)
            for i in range(len(plot_val) - 1):
                cv2.line(frame, (i+plot_x, plot_height-int(plot_val[i])+plot_y), (i+1+plot_x, plot_height-int(plot_val[i+1])+plot_y), plot_color, 1)


        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 32:
            plt.close()
            break


    cv2.destroyAllWindows()


def real_time_hrv(file_path, fs, minBPM, maxBPM):
    is_tracking = False
    track_toler = 1
    prev_bbox = [0, 0, 0, 0]
    detect_th = 0.5
    detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
    tracker = kcf.KCFTracker()

    r_buffer = []
    g_buffer = []
    b_buffer = []

    # 실시간 or 파일 불러와서
    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(file_path)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ##############

    stack = 0
    data_size = 150 # window에 표시할 데이터의 수

    plot_width = int(w*0.7)
    plot_height = int(h*0.15)
    plot_x = int(w/2 - plot_width/2)
    plot_y = int(h*0.8)

    plot_color = (255, 0, 0)
    plot_val = []

    plot_scale = np.linspace(0, plot_width-1, plot_width)

    #past_time = 0
    webcam = False # 프레임레이트 실시간 계산하는거 할거면 True

    init_time=0
    times=[] # hrv계산시 사용할 time data
    ##############

    while True:
        stack += 1

        _, frame = cap.read()

        # hrv에 필요한 time data 취득
        if init_time==0:
            init_time=time.time()
            init_frame=cap.get(cv2.CAP_PROP_POS_MSEC)
        curr_frame=cap.get(cv2.CAP_PROP_POS_MSEC)-init_frame
        times.append(curr_frame) # sec단위로 저장.
        print("[프레임 위치(sec): ",curr_frame*0.001,"]")

        if webcam:
            now_time = time.time()
            if past_time != 0:
                time_interval = now_time-past_time
                fs = int(1/time_interval)
                print(fs)
            past_time = now_time

        if _ == False:
            break



        curr_bbox = []
        # 검출된 얼굴이 존재하지 않는 경우 검출 수행
        if not is_tracking:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False,
                                         True)
            detector.setInput(blob)
            detections = detector.forward()
            bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                      detections[0, 0, i, 2] >= detect_th]
            if len(bboxes) > 0:
                # 가장 큰 얼굴 하나만 검출
                bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
                curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                             bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 트래커에 현재 얼굴 위치 등록
                tracker.init(frame, curr_bbox)
                is_tracking = True

        # 검출된 얼굴이 존재하는 경우 위치 업데이트
        elif is_tracking:
            is_tracking, curr_bbox = tracker.update(frame)
            curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in zip(curr_bbox, prev_bbox)]
            prev_bbox = curr_bbox

        try:
            face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            mask = skinsegment.create_skin_mask(face)
            n_pixel = max(1, np.sum(mask))

            b, g, r = cv2.split(face)
            r[mask == 0] = 0
            g[mask == 0] = 0
            b[mask == 0] = 0

            r = r.astype(np.float32)
            g = g.astype(np.float32)
            b = b.astype(np.float32)

            r_mean = np.sum(r) / n_pixel
            g_mean = np.sum(g) / n_pixel
            b_mean = np.sum(b) / n_pixel

            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)
            # 얼굴에 ROI 그리기
            cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                          (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 2)

        except:
            r_buffer.append(0.0)
            g_buffer.append(0.0)
            b_buffer.append(0.0)

        # 일정한 스택이 쌓이면 실행
        if stack >= data_size:
            # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
            rgb_signal = [r_buffer[-data_size:], g_buffer[-data_size:], b_buffer[-data_size:]]
            f, chrom_spectrum, rPPG = CHROM(rgb_signal, fs, minBPM, maxBPM)
            hrv_features=hrv_analysis(rPPG,times[-data_size:],sr=fs)
            print("[RESULT  time:",times[-1]," HRV:",hrv_features,"]")
            # y업데이트하고 그리기
            minmax = ((rPPG - min(rPPG)) / (max(rPPG) - min(rPPG)))*plot_height
            minmax = interp_1d(minmax, plot_scale) # 플롯된 길이만큼 늘리기

            plot_val.extend(minmax)

            while len(plot_val) > plot_width:
                del plot_val[0]
            # 그래프 창 띄우기
            cv2.rectangle(frame, (plot_x, plot_y, plot_width, plot_height), (255, 255, 255), -1)
            for i in range(len(plot_val) - 1):
                # rppg 그래프그리기
                cv2.line(frame, (i+plot_x, plot_height-int(plot_val[i])+plot_y), (i+1+plot_x, plot_height-int(plot_val[i+1])+plot_y), plot_color, 1)


        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 32:
            plt.close()
            break


    cv2.destroyAllWindows()



def real_time_restore(file_path, fs, minBPM, maxBPM):
    is_tracking = False
    track_toler = 1
    prev_bbox = [0, 0, 0, 0]
    detect_th = 0.5
    detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
    tracker = kcf.KCFTracker()

    r_buffer = []
    g_buffer = []
    b_buffer = []

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(file_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    ##############

    stack = 0
    data_size = 150

    plot_width = int(w * 0.7)
    plot_height = int(h * 0.15)
    plot_x = int(w / 2 - plot_width / 2)
    plot_y = int(h * 0.8)

    plot_color = (255, 0, 0)
    plot_val = []

    plot_scale = np.linspace(0, plot_width - 1, plot_width)

    past_time = 0
    webcam = True

    model_sd = others.model_load('model_rppg/best_model_elu_sd_cubic.h5')
    model_svr = pickle.load(open('model_rppg/SVR_cubic.pkl', 'rb'))

    ##############

    while True:
        stack += 1

        _, frame = cap.read()
        now_time = time.time()
        if webcam:
            if past_time != 0:
                time_inverse = now_time - past_time
                fs = int(1 / time_inverse)
                print(fs)
            past_time = now_time

        if _ == False:
            break

        curr_bbox = []
        # 검출된 얼굴이 존재하지 않는 경우 검출 수행
        if not is_tracking:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False,
                                         True)
            detector.setInput(blob)
            detections = detector.forward()
            bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                      detections[0, 0, i, 2] >= detect_th]
            if len(bboxes) > 0:
                # 가장 큰 얼굴 하나만 검출
                bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
                curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                             bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 트래커에 현재 얼굴 위치 등록
                tracker.init(frame, curr_bbox)
                is_tracking = True

        # 검출된 얼굴이 존재하는 경우 위치 업데이트
        elif is_tracking:
            is_tracking, curr_bbox = tracker.update(frame)
            curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in zip(curr_bbox, prev_bbox)]
            prev_bbox = curr_bbox

        try:
            face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
            mask = skinsegment.create_skin_mask(face)
            n_pixel = max(1, np.sum(mask))

            b, g, r = cv2.split(face)
            r[mask == 0] = 0
            g[mask == 0] = 0
            b[mask == 0] = 0

            r = r.astype(np.float32)
            g = g.astype(np.float32)
            b = b.astype(np.float32)

            r_mean = np.sum(r) / n_pixel
            g_mean = np.sum(g) / n_pixel
            b_mean = np.sum(b) / n_pixel

            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)

            cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                          (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 2)

        except:
            r_buffer.append(0.0)
            g_buffer.append(0.0)
            b_buffer.append(0.0)

        # 일정한 스택이 쌓이면 실행
        if stack >= data_size:
            # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
            rgb_signal = [r_buffer[-data_size:], g_buffer[-data_size:], b_buffer[-data_size:]]
            f, chrom_spectrum, rPPG = CHROM(rgb_signal, fs, minBPM, maxBPM)

            # y업데이트하고 그리기
            minmax = ((rPPG - min(rPPG)) / (max(rPPG) - min(rPPG))) * plot_height
            minmax = interp_1d(minmax, plot_scale)  # 플롯된 길이만큼 늘리기
            rppg_cycle = others.single_cycle_unit_rppg(minmax[17:-17])
            rppg30 = others.feature(rppg_cycle)
            train_rppg_svr = model_svr.predict(np.reshape(rppg30, (-1, 30)))
            train_rppg_sd = model_sd.predict(np.reshape(train_rppg_svr, (-1, 30)))
            restore_rppg = []

            for i in train_rppg_sd:
                restore_rppg.extend(i)

            plot_val.extend(restore_rppg)

            while len(plot_val) > plot_width:
                del plot_val[0]

            cv2.rectangle(frame, (plot_x, plot_y, plot_width, plot_height), (255, 255, 255), -1)
            for i in range(len(plot_val) - 1):
                cv2.line(frame, (i + plot_x, int(plot_height) - int(plot_val[i]) + plot_y),
                         (i + 1 + plot_x, int(plot_height) - int(plot_val[i + 1]) + plot_y), plot_color, 1)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 32:
            plt.close()
            break

    cv2.destroyAllWindows()

