import pickle
import time
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
from hrvanalysis import get_time_domain_features
from scipy.interpolate import interp1d
import datetime
import functions

from uti import kcf
from uti import skinsegment

from scipy.signal import find_peaks


class TotalClass:
    def __init__(self):
        self.restore_rppg_buffer= []

        self.rPPG = []
        self.pulse_rate = 0
        self.model_sd = functions.model_load('model_rppg/best_model_elu_sd_cubic.h5')
        self.model_svr = pickle.load(open('model_rppg/SVR_cubic.pkl', 'rb'))

        self.times=[]

        self.rspo2_buffer = []

    def ppg_restore(self, rppg, plot_height, data_scale):
        restore_rppg = []

        rppg_cycle = functions.single_cycle_unit_rppg(rppg)
        rppg30 = functions.feature(rppg_cycle)

        train_rppg_svr = self.model_svr.predict(np.reshape(rppg30, (-1, 30)))
        train_rppg_sd = self.model_sd.predict(np.reshape(train_rppg_svr, (-1, 30)))

        for i in train_rppg_sd:
            restore_rppg.extend(i)

        minmax = ((restore_rppg - min(restore_rppg)) / (max(restore_rppg) - min(restore_rppg))) * plot_height
        # minmax = for_restore.interp_1d(minmax, data_scale)

        self.restore_rppg_buffer.extend(minmax)

    def hrv_analysis(self, signal, times, sr=30, distance=100):
        #start_time = time.time()
        rppg_sig_ = np.array(signal, dtype='float32')
        rppg_time_ = np.array(times, dtype='float32')  # msec단위
        # ================= preprocessing===================
        # 1. interpolation
        rppg_time = np.linspace(rppg_time_[0], rppg_time_[-1], len(rppg_time_) * 8)  # interpolation된 x
        i = interp1d(rppg_time_, rppg_sig_, kind='quadratic')
        rppg_sig = i(rppg_time)  # interpolation된 y
        # 2. bandpass filtering
        filtered = functions.preprocessing(rppg_sig, 2.0, 0.5, sr)
        r_peaks_y, r_peaks_x = functions.detect_peak_hrv(rppg_time, filtered, distance)

        # 3. extract PPI
        rppg_ppi = np.diff(r_peaks_x)

        # ================ hrv analysis ======================
        hrv_feature = get_time_domain_features(rppg_ppi)  # rppg_ppi r_nni
        #end_time = time.time()
        #print("[hrv 실행시간: ", end_time - start_time, "]") # 반복문 한번당 0.0015
        return hrv_feature

    def rspo2_extract(self, rgb_signal, fs, minBPM, maxBPM):
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

        rspo2 = int(11.8805 * ratio + 79.1914)
        return rspo2


    def real_time_monitoring(self, file_path, fs, minBPM, maxBPM):
        is_tracking = False
        track_toler = 1
        prev_bbox = [0, 0, 0, 0]
        detect_th = 0.5
        detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
        tracker = kcf.KCFTracker()

        r_buffer = []
        g_buffer = []
        b_buffer = []

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #cap = cv2.VideoCapture(file_path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        ##############

        stack = 0
        data_size = 150
        data_scale = np.linspace(0, data_size - 1, data_size)

        plot_width = int(w * 0.6)
        plot_height = int(h * 0.1)
        plot_x = int(w / 2 - plot_width / 2)
        plot_y_restore = int(h * 0.1)
        plot_y_spo2 = int(h * 0.25)
        plot_y_hrv=int(h*0.4)

        plot_color = (255, 0, 0)
        plot_val = []
        plot_rspo2 = []
        plot_hrv = []

        past_time = 0
        webcam = False # True 일 경우 fs 계산

        num = 0
        hrv_init_time=0
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

            # hrv에 필요한 time data 취득
            if stack == 1:
                hrv_init_time = time.time()
            curr_frame = time.time() - hrv_init_time
            self.times.append(curr_frame * 1000)  # sec단위로 저장.
            # print("[프레임 위치(sec): ", curr_frame, "]")

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

            # 원하는 데이터크기 만큼 쌓이면 스레드 진행
            if stack >= data_size:
                # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
                rgb_signal = [r_buffer[-data_size:], g_buffer[-data_size:], b_buffer[-data_size:]]
                _, _, self.rPPG = functions.CHROM(rgb_signal, fs, minBPM, maxBPM)
                # heart rate
                self.pulse_rate = functions.estimate_average_pulserate(self.rPPG, fs, minBPM, maxBPM)
                cv2.putText(frame, 'bpm:%d' % self.pulse_rate, (plot_x, plot_y_restore - int(plot_height / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # 1. hrv
                hrv_features = self.hrv_analysis(self.rPPG, self.times[-data_size:], sr=fs)
                # print("nni_50: ",hrv_features['mean_hr'] )
                # print("pulse", self.pulse_rate)
                plot_hrv.append(hrv_features['nni_50'])

                while len(plot_hrv) > plot_width:
                    del plot_hrv[0]

                cv2.rectangle(frame, (plot_x, plot_y_hrv, plot_width, plot_height), (255, 255, 255), -1)
                for i in range(len(plot_hrv) - 1):
                    cv2.line(frame, (i + plot_x, plot_height - int(plot_hrv[i]) + plot_y_hrv),
                             (i + 1 + plot_x, plot_height - int(plot_hrv[i + 1]) + plot_y_hrv), plot_color, 1)

                #2. spo2
                rspo2_val = self.rspo2_extract(rgb_signal, fs, minBPM, maxBPM)


                self.rspo2_buffer.append(rspo2_val)

                if len(self.rspo2_buffer) >= 30:
                    plot_rspo2.append(int(np.average(self.rspo2_buffer))/100 * plot_height)
                    del self.rspo2_buffer[0]

                while len(plot_rspo2) > plot_width:
                    del plot_rspo2[0]

                cv2.rectangle(frame, (plot_x, plot_y_spo2, plot_width, plot_height), (255, 255, 255), -1)
                for i in range(len(plot_rspo2) - 1):
                    cv2.line(frame, (i + plot_x, plot_height - int(plot_rspo2[i]) + plot_y_spo2),
                             (i + 1 + plot_x, plot_height - int(plot_rspo2[i + 1]) + plot_y_spo2), plot_color, 1)

                # 3. RPPG restore
                if stack % data_size == 0:
                    background_restore = Thread(target=self.ppg_restore, args=(self.rPPG,plot_height,data_scale,))
                    background_restore.start()



            if self.restore_rppg_buffer:
                background_restore.join()
                # num += 1
                # if num == data_size:
                #     num = 0
                plot_val.append(self.restore_rppg_buffer[num])
                num += 1
                # print(len(self.restore_rppg_buffer), -num)

                while len(plot_val) > plot_width:
                    del plot_val[0]

                cv2.rectangle(frame, (plot_x, plot_y_restore, plot_width, plot_height), (255, 255, 255), -1)
                for i in range(len(plot_val) - 1):
                    cv2.line(frame, (i + plot_x, int(plot_height) - int(plot_val[i]) + plot_y_restore),
                             (i + 1 + plot_x, int(plot_height) - int(plot_val[i + 1]) + plot_y_restore), plot_color, 1)


            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == 32:
                plt.close()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'test_sg.mp4'

    fs = 30
    minBPM = 42
    maxBPM = 240
    monitor = TotalClass()
    monitor.real_time_monitoring(video_path, fs, minBPM, maxBPM)

    #
    # restore = RestoreClass()
    #
    # restore.real_time_restore(video_path, fs, minBPM, maxBPM)