import os
import sys
import time
import traceback

from PyQt5 import uic, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import numpy as np
import cv2
import os
import imutils

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('main.ui', self)  # Load the .ui file

        self.FeedLabel = self.findChild(QLabel, "label")

        self.CancelBTN = self.findChild(QPushButton, "pushButton")
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
    NMS_THRESHOLD = 0.3
    MIN_CONFIDENCE = 0.2

    def run(self):
        # Set GPU options
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        start_video = False
        ende_video = False
        zeitanfang = None
        differenz = None
        num = 0
        labelsPath = "coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")
        weights_path = "yolov4-tiny.weights"
        config_path = "yolov4-tiny.cfg"
        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        try:
            self.ThreadActive = True
            Capture = cv2.VideoCapture(0)
            while self.ThreadActive:
                _, frame = Capture.read()
                if _:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # process the frame for pose detection

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
                        if zeitanfang != None and zeitende - zeitanfang >= 0 and differenz == None:
                            differenz = zeitende - zeitanfang

                        if time.time()-zeitanfang >= differenz+30:
                            print("10 seconds passed")
                            zeitanfang = None
                            ende_video = False
                            differenz = None
                        else:
                            videoWriter.write(frame)

                    # Resize image


                    layer_name = model.getLayerNames()
                    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
                    cap = cv2.VideoCapture("streetup.mp4")
                    writer = None

                    image = imutils.resize(frame, width=700)
                    results = self.pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))

                    if results != None:
                        for res in results:
                            cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (255, 160, 122), 2)

                    Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    FlippedImage = cv2.flip(Image, 1)
                    ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)

                    self.ImageUpdate.emit(ConvertToQtFormat)
        except Exception:
            traceback.print_exc()

    def stop(self):
        self.ThreadActive = False
        self.quit()

    def pedestrian_detection(self, image, model, layer_name, personidz=0):
        (H, W) = image.shape[:2]
        results = []

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_name)
        boxes = []
        centroids = []
        confidences = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classID == personidz and confidence > self.MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        idzs = cv2.dnn.NMSBoxes(boxes, confidences, self.MIN_CONFIDENCE, self.NMS_THRESHOLD)
        if len(idzs) > 0:
            for i in idzs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        return results

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())
