import csv
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy
from keras.models import load_model
from scipy.signal import find_peaks

from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate



def avg_filter(ppg_data, N):
    cumsum = np.cumsum(np.insert(ppg_data, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def detect_peak(hrdata, distance, threshold):
    # point[0]가 peak의 x좌표 hrdata[point[0]]이 peak의 y좌표
    point = scipy.signal.find_peaks(hrdata, distance=distance, threshold=threshold)

    # peak = peak 값 (y축값), point[0] = peak의 x위치
    peak = np.zeros(len(point[0]))
    for i in range(len(point[0])):
        peak[i] = (hrdata[point[0][i]])
    return (peak, point[0])


def normalization(ppg_data):
    # MinMaxScaler객체 생성
    scaler = MinMaxScaler()
    # MinMaxScaler 로 데이터 셋 변환. fit() 과 transform() 호출.
    scaler.fit(ppg_data.reshape(len(ppg_data), 1))
    ppg_scaled = scaler.transform(ppg_data.reshape(len(ppg_data), 1))
    # plt.plot(ppg_scaled)

    return ppg_scaled.reshape(len(ppg_data))


def single_cycle_unit_rppg(rppg):
    rppg_data = []

    rppg_inverse = np.dot(rppg, -1)
    rpeaks, rpoints = detect_peak(np.array(rppg_inverse), 15, (0, 29))

    for i in range(len(rpoints) - 1):
        rppg_data.append(rppg[rpoints[i]:rpoints[i + 1]])

    return rppg_data


def single_cycle_unit_cppg(cppg, n):
    cppg_data = []

    cppg0 = avg_filter(cppg, n)
    cppg1 = cppg[n // 2:-n // 2]

    cpeaks, cpoints = detect_peak(np.array(cppg0), n // 2, None)
    for i in range(len(cpoints) - 1):
        cppg_data.append(cppg1[cpoints[i]:cpoints[i + 1]])

    #     for i in range(len(cpoints)-1):
    #         plt.figure(figsize=(5, 5))
    #         plt.plot(cppg1[cpoints[i]:cpoints[i+1]])
    return cppg_data


def feature(ppg):
    ppg_data = []
    for i in range(len(ppg)):
        data = []
        ppg[i] = normalization(np.array(ppg[i]))
        point = np.linspace(0, len(ppg[i]) - 1, 30)
        x = np.linspace(0, len(ppg[i]) - 1, len(ppg[i]))
        f = interpolate.interp1d(x, ppg[i])
        for j in range(30):
            data.append(f(point[j]))
        ppg_data.append(data)
    return ppg_data


def detect_peak_plt(hrdata, distance):
    point = scipy.signal.find_peaks(hrdata, distance=distance)
    # point[0]가 peak의 x좌표 hrdata[point[0]]이 peak의 y좌표
    plt.figure(figsize=(40, 5))
    plt.scatter(point[0], hrdata[point[0]], c='r')
    plt.plot(hrdata)

    peak = np.zeros(len(point[0]))
    for i in range(len(point[0])):
        peak[i] = (hrdata[point[0][i]])
        # peak = peak 값 (y축값), point[0] = peak의 x위치
    print(len(peak))


#    return(peak,point[0])

def cosine_similarity(A, B):
    result = []
    for i in range(len(A)):
        result.append(np.dot(A[i], B[i]) / (np.linalg.norm(A[i]) * np.linalg.norm(B[i])))
    #     return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return result, result.index(max(result)), np.mean(result)


def correlation_coefficient(A, B):
    result = []
    for i in range(len(A)):
        result.append(np.corrcoef(A[i], B[i])[0, 1])
    #     return np.corrcoef(A, B)[0, 1]
    return result, result.index(max(result)), np.mean(result)


def PPI(dataA, dataB, distanceA, distanceB):
    pointA, _ = scipy.signal.find_peaks(dataA, distance=distanceA)
    diffA = np.diff(pointA)

    pointB, _ = scipy.signal.find_peaks(dataB, distance=distanceB)
    diffB = np.diff(pointB)

    PPI_avg = sum(abs(diffA - diffB)) / len(diffA)

    return diffA, diffB, PPI_avg

def model_load(path):
    return load_model(path)

def ppg_save(save_path, signal):
    with open(save_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(signal)
        f.close()

def ppg_load(load_path):
    ppg = []
    with open(load_path, 'r', encoding='utf-8') as f:
        rd = csv.reader(f)
        for line in rd:
            ppg = line
        f.close()
    ppg = list(map(float, ppg))
    return ppg