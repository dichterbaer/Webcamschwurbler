import sys
import cv2
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QThread, Signal, Slot
import pyvirtualcam
from pyvirtualcam import PixelFormat
import imageProcessing as ip
import onnxruntime as ort
import argparse


class Worker(QThread):
    def __init__(self, args):
        super().__init__()        
        print("Worker init")
        self.args = args        
        self.filter = 0
        self.running = True
        self.pausing = False
        self.invert = False
        self.idx = 0

    def setFilter(self, filter):
        print(f"Filter: {self.filter}")
        self.filter = filter

    
    def setRunning(self):
        print(f"Running: {self.running}")
        self.running = not self.running


    def setPausing(self):
        print(f"Pausing: {self.pausing}")
        self.pausing = not self.pausing

    def setInvert(self, invert):
        print(f"Invert: {invert}")
        self.invert = invert


    def run(self):
        print(self.args)
        # Open webcam
        cap = cv2.VideoCapture(self.args["camera"], self.args["capture"])
        pref_width = 1280
        pref_height = 720
        pref_fps = 30
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
        cap.set(cv2.CAP_PROP_FPS, pref_fps)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        face_cascade_handle = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        netzle = ort.InferenceSession("onnx/mosaic-9.onnx", providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        first = True
        # Open virtual camera
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR, device=self.args["device"]) as cam:
            print(f'Using virtual camera: {cam.device}')
            while self.running:
                #print('running')
                if True:#not self.pausing:
                    # Read frame from webcam
                    #print('Reading frame from webcam.')
                    ret, frame = cap.read()
                    if not ret:
                        print('Ignoring empty camera frame.')
                        continue
                    #else:
                    #    print('Reading frame from webcam successful.')
                        
                    if self.filter == 0:
                        frame = frame
                    elif self.filter == 1:
                        frame = ip.CannyFull(frame)
                    elif self.filter == 2:
                        frame = ip.CannyBackground(frame, face_cascade_handle, height, width)
                    elif self.filter == 3:
                        frame = ip.Rotate(frame, self.idx, self.invert)
                    elif self.filter == 4:
                        frame = ip.MirrorMiddleX(frame, self.invert)
                    elif self.filter == 5:
                        frame = ip.MirrorMiddleY(frame, self.invert)
                    elif self.filter == 6:
                        frame = ip.InvertColors(frame)
                    elif self.filter == 7:
                        frame = ip.Spackern(frame, netzle=netzle)
                    else:
                        frame = frame
                    #print(f"Filterindex: { self.filter}")

                    # Send to virtual camera
                    if(not first):
                        cam.sleep_until_next_frame()
                        first = False
                    cam.send(frame)


                    # Update loop index
                    self.idx = self.idx + 1

            # Close webcam
            cap.release()



class MyWidget(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        # Create a layout
        self.layout = QtWidgets.QVBoxLayout(self)
        #create camera thread
        self.worker = Worker(args)
        self.workerThread = QtCore.QThread()
        
        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(self.worker.run)
        
        # Create a label
        self.label = QtWidgets.QLabel("Webcamschwurbler")
        self.layout.addWidget(self.label)
        # Create a start button
        self.startButton = QtWidgets.QPushButton("Start")
        self.startButton.clicked.connect(self.worker.run)
        #self.startButton.clicked.connect(self.startClicked)
        self.layout.addWidget(self.startButton)
        # Create a Stop button
        self.bt_stop = QtWidgets.QPushButton("Stop")
        #self.bt_stop.clicked.connect(self.worker.setRunning)
        self.bt_stop.clicked.connect(self.stopClicked)
        self.layout.addWidget(self.bt_stop)
        # Create a Pause button
        self.bt_pause = QtWidgets.QPushButton("Pause")
        #self.bt_pause.clicked.connect(self.worker.setPausing)
        self.bt_pause.clicked.connect(self.pauseClicked)
        self.layout.addWidget(self.bt_pause)
        # Create a listbox
        self.lbx_filter = QtWidgets.QListWidget()
        self.lbx_filter.addItems(['Kein Filter', 'Canny', 'Canny BG', 'Rotate', 'MirrorX', 'MirrorY', 'InvertColors', 'Mosaic'])
        #connect listbox to worker. setFilter function 
        self.lbx_filter.itemClicked.connect(self.listboxClicked)

        #self.lbx_filter.itemClicked.connect(self.worker.setFilter(self.lbx_filter.currentRow()))
        self.layout.addWidget(self.lbx_filter)
        # Create a checkbox
        self.cbx_invert = QtWidgets.QCheckBox("Invert")
        self.cbx_invert.stateChanged.connect(self.invertClicked)
        self.layout.addWidget(self.cbx_invert)

        #start thread
        self.workerThread.start()

    def invertClicked(self):
        print("invert clicked")
        self.worker.setInvert(self.cbx_invert.isChecked())

    def pauseClicked(self):
        print("pause clicked")
        self.worker.setPausing()
    
    def stopClicked(self):
        print("stop clicked")
        self.worker.setRunning()

    def listboxClicked(self):
        print("listbox clicked")
        self.worker.setFilter(self.lbx_filter.currentRow())

    def closeEvent(self, event):
        self.worker.setRunning()
        self.workerThread.quit()
        self.workerThread.wait()
        event.accept()

    
        



def camSetup(videoCapture):
    pref_width = 1280
    pref_height = 720
    pref_fps = 30
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    videoCapture.set(cv2.CAP_PROP_FPS, pref_fps)


def cameraLoop():
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, help="The loopback-device e.g. /dev/video4")
    parser.add_argument('--capture', default=0, help="OpenCV capture backend to use 0: Devault/Any, see: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d for values")
    parser.add_argument('--camera', default=0, help="Camera ID to use")
    args = vars(parser.parse_args())
    print(args)
    app = QtWidgets.QApplication(sys.argv)
    ex = MyWidget(args)
    ex.show()
    sys.exit(app.exec())   