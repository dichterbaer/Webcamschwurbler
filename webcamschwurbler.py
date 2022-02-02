import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np
from tkinter import *


filter = 0 # Global filter index
running = True # Global flag
idx = 0  # loop index

def stop():
    """Stop scanning by setting the global flag to False."""
    global running
    running = False


def CannyFull(frame):
    canny = cv2.Canny(frame, 100, 200)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return canny


def CannyBackground(frame, cascadeHandle, height, width):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # detect people in the image
    # returns the bounding boxes for the detected objects
    faces = cascadeHandle.detectMultiScale(gray, 1.1, 4)
    canny = cv2.Canny(frame, 100, 200)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    rois = []
    for (x, y, w, h) in faces:
        hdiff = height - y
        wdiff = width - x
        xOff = max(x-50, 0)
        yOff = max(0,y-100)
        
        ROI = frame[yOff:y+hdiff, xOff:x+w+50]
        rois.append(ROI)
        canny[yOff:y+hdiff, xOff:x+w+50]= ROI
    return canny


def getElement(event):
  selection = event.widget.curselection()
  index = selection[0]
  value = event.widget.get(index)
  
  global filter
  filter = index
  print(index,' -> ',value)


root = Tk()
root.title("Webcamschwurbler")
root.geometry("500x500")

app = Frame(root)
app.grid()

stop = Button(app, text="Stop", command=stop)

stop.grid()
var2 = StringVar()
var2.set(('Kein Filter','Canny','Canny BG'))
lb = Listbox(root, listvariable=var2)
lb.grid()
lb.bind('<<ListboxSelect>>', getElement) #Select click


vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

pref_width = 1280
pref_height = 720
pref_fps = 30
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Query final capture device values
# (may be different from preferred settings)
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vc.get(cv2.CAP_PROP_FPS)
with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.BGR) as cam:
    print('Virtual camera device: ' + cam.device)
    while True:
        
        #update gui every 10 frames
        if idx % 10 == 0:
            root.update()
        if not running:
            break

        ret, frame = vc.read()

        if filter == 0:
            frame = frame
        elif filter == 1:
            frame = CannyFull(frame)
        elif filter == 2:
            frame = CannyBackground(frame, face_cascade, height, width)
        
        cam.send(frame)
        idx += 1
        cam.sleep_until_next_frame()







        














