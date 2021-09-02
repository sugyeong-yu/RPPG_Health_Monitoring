
import pickle
import sys
import time
from threading import Thread
import csv

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QMessageBox

import functions
import analysis_functions  # ppg 복원, spo2, hrv 구하는 함수들
import get_rgb_function  # 얼굴 detect하고 rgb 추출하는 함수
from uti import kcf
import result_dialog

form_class = uic.loadUiType('monitoring_5.ui')[0]


class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.setupUi(self)
        self.setWindowTitle('Remote Health Monitoring')

        self.disable_buttons(True)
        self.webcam_button.clicked.connect(self.camera_connect_event)
        self.file_button.clicked.connect(self.filebutton_clicked)

        self.start_button.clicked.connect(lambda: self.controllbutton_clicked(self.start_button))
        self.pause_button.clicked.connect(lambda: self.controllbutton_clicked(self.pause_button))
        self.stop_button.clicked.connect(lambda: self.controllbutton_clicked(self.stop_button))

        self.save_button.clicked.connect(self.savebutton_clicked)

        self.video_thread = VideoThread()
        self.video_thread.image_change_signal.connect(self.set_image)
        self.video_thread.image_change_rppg.connect(self.set_rppg)
        self.video_thread.image_change_spo2.connect(self.set_spo2)
        self.video_thread.image_change_hrv.connect(self.set_hrv)
        self.video_thread.message_go.connect(self.set_message)

    @pyqtSlot(str)
    def set_message(self, str):
        QMessageBox.about(
            self, 'Message', str
        )
        self.video_thread.pause = False
        self.video_thread.stop()
        self.webcam_button.setText('Camera open')
        self.disable_buttons(True)
        self.save_button.setDisabled(False)

        dialog = QDialog()
        ui = result_dialog.Ui_Dialog()
        ui.setupUi(dialog, self.video_thread.plot_pulse_rate, self.video_thread.plot_rspo2,
                   self.video_thread.plot_hrv_mean_nni, self.video_thread.plot_hrv_sdnn,
                   self.video_thread.plot_hrv_pnni_50)
        dialog.show()
        dialog.exec_()

    @pyqtSlot(np.ndarray)
    def set_image(self, frame):
        max_width = self.video_frame.width()
        max_height = self.video_frame.height()

        self.video_thread.graph_W = self.rppg_graph.width()
        self.video_thread.graph_H = int(self.rppg_graph.height() * 0.7)
        self.video_thread.hrv_graph_H = int(self.meannn_graph.height() * 0.7)

        h, w, ch = frame.shape
        show_img = frame
        if h > max_height or w > max_width:
            factor = min(max_height / h, max_width / w)
            show_img = cv2.resize(frame, (int(w * factor), int(h * factor)))
        h, w, ch = show_img.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(show_img.data,
                                            w, h, bytes_per_line,
                                            QtGui.QImage.Format_BGR888 if ch == 3 else QtGui.QImage.Format_BGRA8888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.video_frame.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def set_rppg(self, graph_signal):
        max_width = self.rppg_graph.width()
        max_height = self.rppg_graph.height()
        plot_height = self.video_thread.graph_H

        pulse_rate = self.video_thread.plot_pulse_rate[-1]
        pixmap = self.frame_gen(max_width, max_height, plot_height, graph_signal, 'BPM', pulse_rate)

        self.rppg_graph.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def set_spo2(self, graph_signal):
        max_width = self.spo2_graph.width()
        max_height = self.spo2_graph.height()
        plot_height = self.video_thread.graph_H

        val = graph_signal[-1]
        graph_signal = (graph_signal-70)/30 * plot_height

        pixmap = self.frame_gen(max_width, max_height, plot_height, graph_signal, 'Spo2', val)

        self.spo2_graph.setPixmap(pixmap)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def set_hrv(self, graph_signal1, graph_signal2, graph_signal3):
        hrv_plot_height = self.video_thread.hrv_graph_H

        val1 = graph_signal1[-1]
        val2 = graph_signal2[-1]
        val3 = graph_signal3[-1]

        # 정규화
        graph_signal1 = graph_signal1 / hrv_plot_height
        graph_signal2 = graph_signal2 / hrv_plot_height
        graph_signal3 = graph_signal3 / hrv_plot_height

        # 1. meannn
        max_width1 = self.meannn_graph.width()
        max_height1 = self.meannn_graph.height()
        pixmap1 = self.frame_gen(max_width1, max_height1, hrv_plot_height, graph_signal1, 'meanNN', val1)
        self.meannn_graph.setPixmap(pixmap1)

        # 2. sdnn
        max_width2 = self.sdnn_graph.width()
        max_height2 = self.sdnn_graph.height()
        pixmap2 = self.frame_gen(max_width2, max_height2, hrv_plot_height, graph_signal2, 'SDNN', val2)
        self.sdnn_graph.setPixmap(pixmap2)

        # 3. pnni50
        max_width3 = self.pnni50_graph.width()
        max_height3 = self.pnni50_graph.height()
        pixmap3 = self.frame_gen(max_width3, max_height3, hrv_plot_height, graph_signal3, 'pnni50', val3)
        self.pnni50_graph.setPixmap(pixmap3)

    def frame_gen(self, max_width, max_height, plot_h, graph_signal, str, value=None):
        frame = np.zeros((max_height, max_width, 3), np.uint8)

        plot_x = (max_width) / 200
        plot_y = int((max_height - plot_h) / 2)

        for i in range(len(graph_signal) - 1):
            cv2.line(frame, (int(i * plot_x), plot_h - int(graph_signal[i]) + plot_y),
                     (int((i + 1) * plot_x), plot_h - int(graph_signal[i + 1]) + plot_y), (255, 255, 255), 1)
        if value:
            cv2.putText(frame, str + ':%d' % value, (int(max_width - 140), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 100), 1)

        h, w, ch = frame.shape
        show_img = frame
        if h > max_height or w > max_width:
            factor = min(max_height / h, max_width / w)
            show_img = cv2.resize(frame, (int(w * factor), int(h * factor)))
        h, w, ch = show_img.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(show_img.data,
                                            w, h, bytes_per_line,
                                            QtGui.QImage.Format_BGR888 if ch == 3 else QtGui.QImage.Format_BGRA8888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        return pixmap

    def camera_connect_event(self):
        # 웹캠
        if self.video_thread.capture:
            self.video_thread.stop()
            self.webcam_button.setText('Camera open')
            self.disable_buttons(True)

        self.video_thread.file_path = 0
        self.video_thread.plot_val = [0]  # rppg 그래프 값
        self.video_thread.plot_rspo2 = [0]  # spo2 그래프 값
        self.video_thread.plot_hrv_mean_nni = [0]  # hrv 그래프 값 3가지
        self.video_thread.plot_hrv_sdnn = [0]
        self.video_thread.plot_hrv_pnni_50 = [0]
        self.video_thread.plot_pulse_rate = [0]
        self.video_thread.restore_rppg_buffer = []
        self.video_thread.start()
        self.webcam_button.setText('Camera close')
        self.disable_buttons(False)

    def disable_buttons(self, disable):
        self.start_button.setDisabled(disable)  # 클릭이 가능한데 다른애들은 클릭이안되게끔해주는거가 setDisable
        self.pause_button.setDisabled(disable)
        self.stop_button.setDisabled(disable)
        self.save_button.setDisabled(disable)

    def filebutton_clicked(self):
        if self.video_thread.capture:
            self.video_thread.stop()
            self.disable_buttons(True)
        else:
            self.video_play()

    def video_play(self):
        fname = QFileDialog.getOpenFileName(self)
        self.video_thread.file_path = fname[0]
        self.video_frame.setText(self.video_thread.file_path.split('/')[-1])
        self.video_thread.pause = True

        self.video_thread.plot_val = [0]  # rppg 그래프 값
        self.video_thread.plot_rspo2 = [0]  # spo2 그래프 값
        self.video_thread.plot_hrv_mean_nni = [0]  # hrv 그래프 값 3가지
        self.video_thread.plot_hrv_sdnn = [0]
        self.video_thread.plot_hrv_pnni_50 = [0]
        self.video_thread.plot_pulse_rate = [0]
        self.video_thread.restore_rppg_buffer = []
        self.video_thread.start()
        self.disable_buttons(False)
        self.save_button.setDisabled(True)

    def controllbutton_clicked(self, button):
        if button == self.start_button:
            # 분석시작
            self.video_thread.pause = False
            self.video_thread.play = True
        elif button == self.stop_button:
            self.video_thread.pause = False
            self.video_thread.stop()
            self.webcam_button.setText('Camera open')
            self.disable_buttons(True)
            self.save_button.setDisabled(False)

            dialog = QDialog()
            ui = result_dialog.Ui_Dialog()
            ui.setupUi(dialog, self.video_thread.plot_pulse_rate, self.video_thread.plot_rspo2,
                       self.video_thread.plot_hrv_mean_nni, self.video_thread.plot_hrv_sdnn, self.video_thread.plot_hrv_pnni_50)
            dialog.show()
            dialog.exec_()
        else:
            self.video_thread.pause = True

    def savebutton_clicked(self):
        FileSave = QFileDialog.getSaveFileName(self, 'Save file', './', 'csv file (*.csv)')
        file_write = []
        file_write.append(['mean BPM', np.average(self.video_thread.plot_pulse_rate)])
        file_write.append(['mean SPO2', np.average(self.video_thread.plot_rspo2)])
        file_write.append(['mean nni', np.average(self.video_thread.plot_hrv_mean_nni)])
        file_write.append(['mean SDNN', np.average(self.video_thread.plot_hrv_sdnn)])
        file_write.append(['mean PNN_50', np.average(self.video_thread.plot_hrv_pnni_50)])
        a = ['resorted PPG']
        a.extend(self.video_thread.restore_rppg_buffer)
        file_write.append(a)
        a = ['SPO2']
        a.extend(self.video_thread.plot_rspo2)
        file_write.append(a)
        a = ['nni']
        a.extend(self.video_thread.plot_hrv_mean_nni)
        file_write.append(a)
        a = ['SDNN']
        a.extend(self.video_thread.plot_hrv_sdnn)
        file_write.append(a)
        a = ['PNN_50']
        a.extend(self.video_thread.plot_hrv_pnni_50)
        file_write.append(a)

        if FileSave[0]:
            with open(FileSave[0], 'w', newline='') as f:
                wr = csv.writer(f)
                for i in range(len(file_write)):
                    wr.writerow(file_write[i])

                f.close()
            QMessageBox.about(
                self, 'Message', 'File has been saved :)'
            )


class VideoThread(QThread):
    file_path = None
    capture = False
    play = False
    pause = False

    image_change_signal = pyqtSignal(np.ndarray)  # 비디오 프레임 전달
    image_change_rppg = pyqtSignal(np.ndarray)  # rppg 값 전달
    image_change_spo2 = pyqtSignal(np.ndarray)  # spo2 값 전달
    image_change_hrv = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # hrv 값 전달
    message_go = pyqtSignal(str)

    restore_rppg_buffer = []
    background_restore = None
    fs = 30
    minBPM = 42
    maxBPM = 240

    # 그래프 변수
    graph_W = 0
    graph_H = 0
    hrv_graph_H = 0

    # 그래프 변수 & result 변수
    plot_val = [0]  # rppg 그래프 값
    plot_rspo2 = [0]  # spo2 그래프 값
    plot_hrv_mean_nni = [0]  # hrv 그래프 값 3가지
    plot_hrv_sdnn = [0]
    plot_hrv_pnni_50 = [0]
    plot_pulse_rate = [0]

    def run(self):
        print("CAM RUN")
        self.capture = True
        if type(self.file_path) == str:
            cap = cv2.VideoCapture(self.file_path)
        else:
            cap = cv2.VideoCapture(self.file_path, cv2.CAP_DSHOW)

        # 얼굴 detect, rgb 추출 변수
        is_tracking = False  # face tracking 초기값
        prev_bbox = [0, 0, 0, 0]  # face box 초기값
        detector = cv2.dnn.readNetFromTensorflow('model/face_detector.pb', 'model/face_detector.pbtxt')
        tracker = kcf.KCFTracker()
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        r_buffer = []
        g_buffer = []
        b_buffer = []

        # 그래프 변수
        num = 0  # rppg 그래프 그릴때 변수

        # rppg_restore 변수
        model_sd = functions.model_load('model_rppg/best_model_elu_sd_cubic.h5')
        model_svr = pickle.load(open('model_rppg/SVR_cubic.pkl', 'rb'))

        # spo2 변수
        rspo2_buffer = []

        # hrv 변수
        times = []
        hrv_init_time = 0

        ##그외 변수
        stack = 0  # 데이터 쌓임 정도
        data_size = 150  # 필요한 최소 데이터 개수
        past_time = 0  # fs를 계산하기 위한 초기값
        webcam = False  # True 일 경우 fs 계산

        while self.capture:

            # plot_width = self.graph_W
            plot_height = self.graph_H
            plot_width = 200

            ret, frame = cap.read()

            if not ret:
                self.message_go.emit('Video is finished! ㄴ(^0^)ㄱ')
                break
            if self.pause:
                while True:
                    if self.pause == False:
                        break
            if self.play:
                stack += 1
                # 실시간 fs 추출
                now_time = time.time()
                if webcam:
                    if past_time != 0:
                        time_inverse = now_time - past_time
                        self.fs = int(1 / time_inverse)
                        print(self.fs)
                    past_time = now_time

                # 프레임에서 rgb 신호 취득
                is_tracking = get_rgb_function.get_rgb(frame, is_tracking, tracker, detector, w, h, prev_bbox, r_buffer, g_buffer,
                                         b_buffer)

                if len(r_buffer) > data_size:
                    del r_buffer[0]
                    del g_buffer[0]
                    del b_buffer[0]

                if r_buffer.count(0) >100:
                    self.message_go.emit('Where is your face? `0`')
                    break

                # hrv에 필요한 time data 취득
                if stack == 1:
                    hrv_init_time = time.time()
                curr_frame = time.time() - hrv_init_time
                times.append(curr_frame * 1000)  # sec단위로 저장.

                # 원하는 data_size 만큼 쌓이면 분석 진행/ rppg tread 진행
                if stack >= data_size:
                    # 신호 원하는 데이터 길이 만큼만 가져와서 필터링
                    rgb_signal = [r_buffer, g_buffer, b_buffer]
                    rPPG = functions.CHROM(rgb_signal, self.fs, self.minBPM, self.maxBPM)

                    # heart rate
                    pulse_rate_val = analysis_functions.estimate_average_pulserate(rPPG, self.fs, self.minBPM,
                                                                                    self.maxBPM)
                    self.plot_pulse_rate.append(pulse_rate_val)

                    # 1. RPPG restore
                    if stack % data_size == 0:
                        self.background_restore = Thread(target=analysis_functions.ppg_restore,
                                                    args=(rPPG, plot_height, model_svr, model_sd,
                                                          self.restore_rppg_buffer,))
                        self.background_restore.start()

                    # 2. spo2
                    rspo2_val = analysis_functions.rspo2_extract(rgb_signal, self.fs, self.minBPM, self.maxBPM,
                                                                 rspo2_buffer)
                    if rspo2_val is not None:
                        self.plot_rspo2.append(rspo2_val)

                    # 3. hrv
                    hrv_features = analysis_functions.hrv_analysis(rPPG, times[-data_size:], sr=self.fs)
                    self.plot_hrv_mean_nni.append(hrv_features['mean_nni'])
                    self.plot_hrv_sdnn.append(hrv_features['sdnn'])
                    self.plot_hrv_pnni_50.append(hrv_features['pnni_50'])

                    self.background_restore.join()

                if self.restore_rppg_buffer:
                    self.plot_val.append(self.restore_rppg_buffer[num])
                    num += 1
                    while len(self.plot_val) > plot_width:
                        del self.plot_val[0]

            self.image_change_hrv.emit(np.array(self.plot_hrv_mean_nni[-plot_width:]), np.array(self.plot_hrv_sdnn[-plot_width:]), np.array(self.plot_hrv_pnni_50[-plot_width:]))
            self.image_change_spo2.emit(np.array(self.plot_rspo2[-plot_width:]))
            self.image_change_rppg.emit(np.array(self.plot_val[-plot_width:]))
            self.image_change_signal.emit(frame)

    def stop(self):
        print('stop')
        self.capture = False
        self.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()