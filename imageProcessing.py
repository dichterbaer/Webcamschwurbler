import cv2
import numpy as np

def CannyRainbowPuke(frame, phaseshift = np.pi/4, frequency = 1, rotation_angle = 0, sensitivity = 50):
    sensitivity = 1 - (sensitivity / 100)
    tr2 = 255 * sensitivity
    tr1 = tr2 * 0.4
    canny = cv2.Canny(frame, tr1, tr2)
    width = frame.shape[1]
    height = frame.shape[0]
    size = np.min([width, height])
    x = np.linspace(0, frequency * np.pi, size)
    y = np.linspace(0, frequency * np.pi, size)
    # scale the x and y values to the size of the image
    theta = np.radians(rotation_angle)
    X, Y = np.meshgrid(x, y)
    X = X * np.cos(theta) - Y * np.sin(theta)
    Y = X * np.sin(theta) + Y * np.cos(theta)
    distance = np.sqrt(X**2 + Y**2)
    gray_values = (np.sin(distance + phaseshift) + 1) / 2  # Add the offset to the sine wave
    gray_values = gray_values * 255 
    gray_image = gray_values.astype(np.uint8)
    # scale the gray image to the size of the canny image
    gray_image = cv2.resize(gray_image, (width, height))
    rainbow_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_RAINBOW)
    # Combine the canny image with the rainbow image
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    canny = np.where(canny > 0, rainbow_image, canny)

    return canny


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


def MirrorMiddleY(frame, invert):
    dims = frame.shape
    height = dims[0]
    width = dims[1]
    flipped = cv2.flip(frame, 1)
    if invert: 
        frame[0:height, int(width/2):width] = flipped[0:height, int(width/2):width]
    else:
        frame[0:height, 0:int(width/2)] = flipped[0:height, 0:int(width/2)]
    return frame 


def MirrorMiddleX(frame, invert):
    dims = frame.shape
    height = dims[0]
    width = dims[1]
    flipped = cv2.flip(frame, 0)
    if invert: 
        frame[int(height/2):height, 0:width] = flipped[int(height/2):height, 0:width]
    else:
        frame[0:int(height/2), 0:width] = flipped[0:int(height/2), 0:width]
    return frame 


def Rotate(image, index, invert, speed):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    index = index*speed
    angle = -index%360.0 if invert else index%360.0
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def InvertColors(image):
    image = cv2.bitwise_not(image)
    return image


def MirrorImageX(image):
    flipped = cv2.flip(image, 0)
    image = flipped
    return image


def MirrorImageY(image):
    flipped = cv2.flip(image, 1)
    image = flipped
    return image


def Spackern(image, netzle):
    shp = np.min(image.shape[0:2])
    shp2 = np.max(image.shape[0:2])
   
    imgggg = np.zeros(image.shape, dtype=np.uint8)
    
    offi = (shp2-shp)//2
    image_cut = image[0:shp, offi:offi+shp,:].astype(np.float32)
    image_cut = cv2.GaussianBlur(image_cut, (0,0), sigmaX=3)
    image_cut = cv2.resize(image_cut, (224,224))    
    image_cut = cv2.cvtColor(image_cut, cv2.COLOR_BGR2RGB)    
    image_cut = np.swapaxes(image_cut, 0, 2)
    asbest = np.expand_dims(image_cut,axis=0)
    
    ret = netzle.run(None, {'input1': asbest})
    imgg = ret[0][0]
    imgg = np.swapaxes(imgg, 0, 2) 
    imgg = cv2.cvtColor(imgg, cv2.COLOR_RGB2BGR)    
    imgg = cv2.resize(imgg, (shp,shp))
    imgg = imgg.astype(np.uint8)

    imgggg[0:shp, offi:offi+shp,:] = imgg
    return imgggg
