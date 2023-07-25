import os
import sys
import time
import traceback

from PyQt5 import uic, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import mediapipe as mp



class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('main.ui', self)  # Load the .ui file

        self.FeedLabel = self.findChild(QLabel, "label")


        self.CancelBTN  = self.findChild(QPushButton, "pushButton")
        self.CancelBTN.clicked.connect(self.CancelFeed)


        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)


    def ImageUpdateSlot(self, Image):
        Pic = Image.scaled(int(self.FeedLabel.width()), int(self.FeedLabel.height()), QtCore.Qt.KeepAspectRatio)
        self.FeedLabel.setPixmap(QPixmap(Pic))


    def CancelFeed(self):
        self.Worker1.stop()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        ## initialize pose estimator
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        start_video = False
        ende_video = False
        zeitanfang = None
        num = 0
        try:
            self.ThreadActive = True
            #Capture = cv2.VideoCapture(0)
            Capture = cv2.VideoCapture('rtsp://admin:admin@192.168.178.48:554')  # IP Camera
            while self.ThreadActive:
                _, frame = Capture.read()
                if _:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # process the frame for pose detection
                    pose_results = pose.process(frame_rgb)
                    if start_video:
                        if ende_video == False:
                            zeitanfang = time.time()
                            ende_video = True
                            print("video is recording")
                            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                            videoWriter = cv2.VideoWriter('./videos/video'+str(num)+'.avi', fourcc, 30.0, (640,480))
                            num += 1
                        videoWriter.write(frame)
                    elif start_video == False and ende_video == True:
                        zeitende = time.time()
                        if zeitanfang != None and zeitende - zeitanfang >= 0:
                            print(zeitende - zeitanfang)
                            videoWriter.release()
                            zeitanfang = None
                            ende_video = False
                    if pose_results.pose_landmarks != None and start_video == False:
                        print("person detected")
                        start_video = True
                    elif pose_results.pose_landmarks == None:
                        start_video = False

                    # print(pose_results.pose_landmarks)

                    # draw skeleton on the frame
                    mp_drawing.draw_landmarks(
                        frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )

                    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)

                    self.ImageUpdate.emit(ConvertToQtFormat)
        except Exception:
            traceback.print_exc()
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())