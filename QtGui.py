import sys
import cv2
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QThread, Signal, Slot
import pyvirtualcam
from pyvirtualcam import PixelFormat
import imageProcessing as ip
import onnxruntime as ort
import argparse
import numpy as np


class Worker(QThread):
    def __init__(self, args, labbl):
        super().__init__()        
        print("Worker init")
        self.args = args        
        self.filter = 0
        self.running = True
        self.pausing = False
        self.invert = False
        self.idx = 0
        self.speed = 0
        self.speed2 = 0
        self.imageLabel = labbl
        self.sensitivity = 0

    def setFilter(self, filter):
        print(f"Filter: {self.filter}")
        self.filter = filter

    
    def setRunning(self):
        self.pausing = False
        self.running = True
        
    def setStopping(self):        
        self.running = False
        


    def setPausing(self):
        print(f"Pausing: {self.pausing}")
        self.pausing = not self.pausing

    def setInvert(self, invert):
        print(f"Invert: {invert}")
        self.invert = invert
        
    def setSpeed(self, speed):        
        self.speed = speed

    def setSpeed2(self, speed2):
        self.speed2 = speed2

    def calcPhase(self, idx, invert):
        idx = idx*self.speed / 100
        return -idx%(2*np.pi) if invert else idx%(2*np.pi)
    
    def calcRotationAngle(self, idx, invert):
        idx = idx*self.speed2
        return -idx%360.0 if invert else idx%360.0
    
    def setSensitivity(self, sensitivity):
        self.sensitivity = sensitivity


    def run(self):
      
        # Open webcam
        cap = cv2.VideoCapture(self.args["camera"], int(self.args["capture"]))

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
        netzle = ort.InferenceSession("onnx/mosaic-9.onnx", providers=self.args["inferenceBackends"])
        first = True
        # Open virtual camera
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR, device=self.args["device"]) as cam:
            print(f'Using virtual camera: {cam.device}')
            while self.running:
                #print('running')
                if self.pausing:
                    self.msleep(20)
                else:
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
                        frame = ip.Rotate(frame, self.idx, self.invert, self.speed)
                    elif self.filter == 4:
                        frame = ip.MirrorMiddleX(frame, self.invert)
                    elif self.filter == 5:
                        frame = ip.MirrorMiddleY(frame, self.invert)
                    elif self.filter == 6:
                        frame = ip.InvertColors(frame)
                    elif self.filter == 7:
                        frame = ip.Spackern(frame, netzle=netzle)
                    elif self.filter == 8:
                        phase = self.calcPhase(self.idx, self.invert)
                        rotationAngle = self.calcRotationAngle(self.idx, self.invert)
                        frame = ip.CannyRainbowPuke(frame, phase, 1, rotationAngle, self.sensitivity)
                    else:
                        frame = frame
                    #print(f"Filterindex: { self.filter}")

                    # Send to virtual camera
                    if(not first):
                        cam.sleep_until_next_frame()
                        first = False
                    cam.send(frame)
                    
                    
                    h, w = frame.shape[:2]
                    bytesPerLine = 3 * w
                    qimage = QtGui.QImage(frame.data, w, h, bytesPerLine, QtGui.QImage.Format.Format_BGR888) 
                    q_pixmap = QtGui.QPixmap.fromImage(qimage)
                    self.imageLabel.setPixmap(q_pixmap)
                        
                    # Update loop index
                    self.idx = self.idx + 1

            # Close webcam
            cap.release()


class MyWidget(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Create a layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.imageLabel = QtWidgets.QLabel()
        self.worker = Worker(self.args, self.imageLabel)
       
        
        # Create a label
        self.label = QtWidgets.QLabel("Webcamschwurbler")
        self.layout.addWidget(self.label)
        # Create a start button
        self.startButton = QtWidgets.QPushButton("Start")
        self.startButton.clicked.connect(self.startWorker)
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
        self.lbx_filter.addItems(['Kein Filter', 'Canny', 'Canny BG', 'Rotate', 'MirrorX', 'MirrorY', 'InvertColors', 'Mosaic', 'RainbowPuke'])
        #connect listbox to worker. setFilter function 
        self.lbx_filter.itemClicked.connect(self.listboxClicked)

        #self.lbx_filter.itemClicked.connect(self.worker.setFilter(self.lbx_filter.currentRow()))
        self.layout.addWidget(self.lbx_filter)
        # Create a checkbox
        self.cbx_invert = QtWidgets.QCheckBox("Invert")
        self.cbx_invert.stateChanged.connect(self.invertClicked)
        self.sl_rotSpeed = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.sl_rotSpeed.setMinimum(0)
        self.sl_rotSpeed.setMaximum(1000)
        self.sl_rotSpeed.valueChanged.connect(self.rotSpeedChanged)        
        self.layout.addWidget(self.cbx_invert)
        self.layout.addWidget(self.sl_rotSpeed)

        # add another speed slider
        self.sl_speed = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.sl_speed.setMinimum(0)
        self.sl_speed.setMaximum(1000)
        self.layout.addWidget(self.sl_speed)
        self.sl_speed.valueChanged.connect(self.speedChanged) 

        # add sensitivity slider
        self.sl_sensitivity = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.sl_sensitivity.setMinimum(0)
        self.sl_sensitivity.setMaximum(100)
        self.layout.addWidget(self.sl_sensitivity)
        self.sl_sensitivity.valueChanged.connect(self.sensitivityChanged)


        
        
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)        
        self.layout.addWidget(self.imageLabel)
        
        self.rotSpeedChanged()
        self.speedChanged()
        
        self.startWorker()
        
        
    def startWorker(self):
        self.worker.setRunning()
        self.worker.start()

    def invertClicked(self):
        print("invert clicked")
        self.worker.setInvert(self.cbx_invert.isChecked())
        
    def rotSpeedChanged(self):        
        self.worker.setSpeed(self.sl_rotSpeed.value()*0.04)

    def speedChanged(self):        
        self.worker.setSpeed2(self.sl_speed.value()*0.04)

    def sensitivityChanged(self):
        self.worker.setSensitivity(self.sl_sensitivity.value())

    def pauseClicked(self):
        print("pause clicked")
        self.worker.setPausing()
    
    def stopClicked(self):
        print("stop clicked")
        self.worker.setStopping()
        

    def listboxClicked(self):
        print("listbox clicked")
        self.worker.setFilter(self.lbx_filter.currentRow())

    def closeEvent(self, event):
        self.worker.setStopping()
        self.worker.quit()
        self.worker.wait()
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
    parser.add_argument('--capture', default=700, help="OpenCV capture backend to use 0: Devault/Any, 700: DShow, see: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ga023786be1ee68a9105bf2e48c700294d for values")
    parser.add_argument('--camera', default=0, help="Camera ID to use")
    parser.add_argument('--inferenceBackends', '--names-list', nargs='+', default=['CUDAExecutionProvider','CPUExecutionProvider'], help="Execution providers to use, call: --inferenceBackends CPUExecutionProvider CUDAEcecutionProvider")
    args = vars(parser.parse_args())
    print(args)   
    app = QtWidgets.QApplication(sys.argv)
    ex = MyWidget(args)
    ex.show()
    sys.exit(app.exec())   