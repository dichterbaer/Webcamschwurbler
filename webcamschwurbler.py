import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np
import imageProcessing as ip
from tkinter import *


filter = 0 # Global filter index
running = True # Global running flag
pausing = False # Global pause flag
idx = 0  # loop index

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports

def stop():
    """Stop scanning by setting the global flag to False."""
    vc.release()
    global running
    running = False


def pause():
    """Stop scanning by setting the global flag to False."""
    global pausing
    pausing = not pausing
    if pausing:
        pause.config(text="Resume") 
    else:
        pause.config(text="Pause")


def getElement(event):
  selection = event.widget.curselection()
  index = selection[0]
  value = event.widget.get(index)
  
  global filter
  filter = index
  print(index,' -> ',value)

#list_ports()

root = Tk()
root.title("Webcamschwurbler")
root.geometry("250x250")

app = Frame(root)
app.grid()

stop = Button(app, text="Stop", command=stop)
stop.grid(row=0, column=0)

pause = Button(app, text="Pause", command=pause, activebackground='orange')
pause.grid(row=0, column=1)

var2 = StringVar()
var2.set(('Kein Filter', 'Canny', 'Canny BG', 'Rotate', 'MirrorX', 'MirrorY', 'InvertColors'))
lb = Listbox(root, listvariable=var2)
lb.grid(row=1, column=0, columnspan=2)
lb.bind('<<ListboxSelect>>', getElement) #Select click

invert = IntVar()
mirrorImgX = IntVar()
mirrorImgY = IntVar()
cb_invert = Checkbutton(text='Invert Filter', variable=invert)
cb_invert.grid(row=1, column=3)
cb_mirrorImgX = Checkbutton(text='Mirror X', variable=mirrorImgX)
cb_mirrorImgX.grid(row=2, column=3)
cb_mirrorImgY = Checkbutton(text='Mirror Y', variable=mirrorImgY)
cb_mirrorImgY.grid(row=3, column=3)

vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not vc.isOpened():
    raise RuntimeError('Could not open video source')

pref_width = 1280
pref_height = 720
pref_fps = 30
vc.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
vc.set(cv2.CAP_PROP_FPS, pref_fps)

face_cascade_handle = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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
            frame = ip.CannyFull(frame)
        elif filter == 2:
            frame = ip.CannyBackground(frame, face_cascade_handle, height, width)
        elif filter == 3:
            frame = ip.Rotate(frame, idx, invert.get())
            frame = frame
        elif filter == 4:
            frame = ip.MirrorMiddleX(frame, invert.get())
        elif filter == 5:
            frame = ip.MirrorMiddleY(frame, invert.get())
        elif filter == 6:
            frame = ip.InvertColors(frame)
        
        if mirrorImgX.get():
            frame = ip.MirrorImageX(frame)
        if mirrorImgY.get():
            frame = ip.MirrorImageY(frame)

        cam.send(frame)
        idx += 1
        cam.sleep_until_next_frame()







        














