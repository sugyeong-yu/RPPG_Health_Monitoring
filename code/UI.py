import sys

import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

form_class = uic.loadUiType('monitoring.ui')[0]
class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.setupUi(self)
        self.disable_buttons(True)
        self.webcam_button.clicked.connect(self.camera_connect_event)
        self.file_button.clicked.connect(self.filebutton_clicked)
        self.start_button.clicked.connect(lambda: self.controllbutton_clicked(self.start_button))
        self.pause_button.clicked.connect(lambda: self.controllbutton_clicked(self.pause_button))
        self.stop_button.clicked.connect(lambda: self.controllbutton_clicked(self.stop_button))

        self.video_thread = VideoThread()
        self.video_thread.image_change_signal.connect(self.set_image)

    @pyqtSlot(np.ndarray)
    def set_image(self, frame):
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap(image)
        self.label_image.setPixmap(pixmap)

    def camera_connect_event(self):
        # 웹캠
        if self.video_thread.capture:
            self.video_thread.stop()
            self.webcam_button.setText('Camera open')
            self.disable_buttons(True)
        else:
            self.video_thread.file_path=0
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
        self.video_thread.pause = True
        self.video_thread.start()
        self.disable_buttons(False)

    def controllbutton_clicked(self,button):
        if button == self.start_button:
            # 분석시작
            self.video_thread.pause=False
            self.video_thread.play=True
        elif button == self.stop_button:
            self.video_thread.pause = False
            self.video_thread.stop()
            self.webcam_button.setText('Camera open')
        else:
            self.video_thread.pause = True


class VideoThread(QThread):
    image_change_signal = pyqtSignal(np.ndarray)
    file_path=None
    capture = False
    play=False
    pause=False
    def run(self):
        print("CAM RUN")
        self.capture = True
        if type(self.file_path) == str :
            cap = cv2.VideoCapture(self.file_path)
        else:
            cap = cv2.VideoCapture(self.file_path,cv2.CAP_DSHOW)
        i=0 # test용
        while self.capture:
            ret, frame = cap.read()
            print(ret)
            if not ret:
                print("could not connet camera")
                break
            if self.pause:
                while True:
                    if self.pause == False:
                        break
            if self.play:
                i += 1
                print(i)
                # rppg 프로그램 실행
            self.image_change_signal.emit(frame)

    def stop(self):
        print('stop')
        self.capture = False

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()