import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np
from tkinter import *


filter = 0 # Global filter index
running = True # Global running flag
pausing = False # Global pause flag
idx = 0  # loop index

def stop():
    """Stop scanning by setting the global flag to False."""
    global running
    running = False


def pause():
    """Stop scanning by setting the global flag to False."""
    global pausing
    pausing = not pausing


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


def FourierJet(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)
    rows, cols = frame.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return img_back


def MirrorY(frame, invert):
    dims = frame.shape
    height = dims[0]
    width = dims[1]
    flipped = cv2.flip(frame, 1)
    if invert: 
        frame[0:height, int(width/2):width] = flipped[0:height, int(width/2):width]
    else:
        frame[0:height, 0:int(width/2)] = flipped[0:height, 0:int(width/2)]
    return frame 


def MirrorX(frame, invert):
    dims = frame.shape
    height = dims[0]
    width = dims[1]
    flipped = cv2.flip(frame, 0)
    if invert: 
        frame[int(height/2):height, 0:width] = flipped[int(height/2):height, 0:width]
    else:
        frame[0:int(height/2), 0:width] = flipped[0:int(height/2), 0:width]
    return frame 

    



def Rotate(image, index, invert):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    angle = -index%360 if invert else index%360
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def getElement(event):
  selection = event.widget.curselection()
  index = selection[0]
  value = event.widget.get(index)
  
  global filter
  filter = index
  print(index,' -> ',value)


root = Tk()
root.title("Webcamschwurbler")
root.geometry("180x195")

app = Frame(root)
app.grid()

stop = Button(app, text="Stop", command=stop)
stop.grid(row=0, column=0)

pause = Button(app, text="pause", command=pause, activebackground='orange')
pause.grid(row=0, column=1)

var2 = StringVar()
var2.set(('Kein Filter', 'Canny', 'Canny BG', 'Rotate', 'MirrorX', 'MirrorY'))
lb = Listbox(root, listvariable=var2)
lb.grid(row=1, column=0, columnspan=2)
lb.bind('<<ListboxSelect>>', getElement) #Select click

invert = IntVar()
cb_invert = Checkbutton(text='invert', variable=invert)
cb_invert.grid(row=1, column=3)

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
        if idx % 1 == 0:
            root.update()
        if not running:
            break
        if pausing:
            idx += 1
            continue
        
        ret, frame = vc.read()

        if filter == 0:
            frame = frame
        elif filter == 1:
            frame = CannyFull(frame)
        elif filter == 2:
            frame = CannyBackground(frame, face_cascade, height, width)
        elif filter == 3:
            frame = Rotate(frame, idx, invert.get())
            frame = frame
        elif filter == 4:
            frame = MirrorX(frame, invert.get())
        elif filter == 5:
            frame = MirrorY(frame, invert.get())
        
        cam.send(frame)
        idx += 1
        cam.sleep_until_next_frame()







        














