
# coding: utf-8

# This notebook requred installed tesseract engine. Instruction about installation of the engine i here:
# https://github.com/tesseract-ocr/tesseract
# 
# To start: Run All
# To exit: press "Q"
# Main window contains wideo streem. "Crop" windows dislay filters work.
# Recognised ivent shown on main windows after red "Text:".

# In[1]:


import cv2
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
from PIL import Image
import pytesseract
import os


# In[2]:


def setAnalisisArea(w,h):
    streamWIDTH  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    streamHEIGHT = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)


    # Define an initial bounding box
    abox = (int(w),int(h))
    bbox = (streamWIDTH/2-abox[0]/2, streamHEIGHT*2/3-abox[1]/2,abox[0], abox[1])
    
    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1,p2


# In[3]:


def drawInfoBox(frame,p1,p2,text,ocr_text):
    
    # Check screen parameters
    streamFPS = stream.get(cv2.CAP_PROP_FPS)
    streamWIDTH  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    streamHEIGHT = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Draw bounding box
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)


    # Display text on frame
    cv2.putText(frame,"Info: ", (p2[0]+10,p1[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);  
    cv2.putText(frame,"OCR:    "+ocr_text, (p2[0]+10,p1[1] + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);  

#    cv2.putText(frame,"FPS:    "+str(streamFPS), (p2[0]+10,p1[1] + 15), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);  
#    cv2.putText(frame,"Width:  "+str(streamWIDTH), (p2[0]+10,p1[1] + 40), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);  
#    cv2.putText(frame,"Height: "+str(streamHEIGHT), (p2[0]+10,p1[1] + 65), 
#                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2);  
    cv2.putText(frame,"Text: "+ text, (p2[0]+10,p1[1] + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2);  


# In[4]:


def extractText(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Red lower mask (0-10)
    lower_red = np.array([0,70,50])
#    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask_r0 = cv2.inRange(hsv, lower_red, upper_red)

    # Red upper mask (170-180)
    lower_red = np.array([170,70,50])
#    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask_r1 = cv2.inRange(hsv, lower_red, upper_red)


    # define range of white color in HSV
    lower_white = np.array([0,0,230])
    upper_white = np.array([180,20,255])
    mask_w = cv2.inRange(hsv, lower_white, upper_white)

    # join my masks
#    mask_2 = cv2.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
#    mask = mask +mask0+mask1
#    mask = mask + mask1
#    mask = cv2.bitwise_xor(mask1, mask)
#    mask = mask_r0|mask_r1|mask_w
    mask = mask_w

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
 
    return res


# In[5]:


VIDEO_URL = "https://bl.webcaster.pro/media/playlist/free_361ac6c7ea479629f777e6fab966ead2_hd/381_81890575/1080p/74bd89dc811014cb7e81b42b51074713/4682340436.m3u8"

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
stream = cv2.VideoCapture(VIDEO_URL)


# Check if camera opened successfully
if (stream.isOpened()== False): 
    print("Error opening video stream or file")


# In[6]:


from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

w,h = 480,240
# Read until video is completed
while(stream.isOpened()):
    # Capture frame-by-frame
    ret, frame = stream.read()
    if ret == True:
        p1,p2 = setAnalisisArea(w,h)
        
        crop_img = frame[p1[1]:p2[1],p1[0]:p2[0]]
        
        img = extractText(crop_img)

#        cv2.imshow('Crop',img)        
#        createTrackBar('Crop')
        # Convert it into grayscale and display again
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bitwise_not(gray)
        cv2.imshow('Crop',gray)        
        
        # Smooth the image to get more accurate results
        # Define the kernel size for gaussian smoothing
#        kernel_size = 5
#        blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
#        cv2.imshow('Crop', blur_gray) 

#        filename = "{}.png".format(os.getpid())
#        cv2.imwrite(filename, gray)
#        text = pytesseract.image_to_string(Image.open(filename))
#        os.remove(filename)
        
        # OR explicit beforehand converting
        ocr_text = pytesseract.image_to_string(Image.fromarray(gray))
#        print(pytesseract.image_to_string(Image.fromarray(gray))
#        print(ocr_text)
#        low_threshold = 100
#        high_threshold = 200
#        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
#        cv2.imshow('Crop', edges) 

        if "eliminated" in ocr_text.lower():
            text = "Eliminated event detected"
        elif "asist" in ocr_text.lower():
            text = "Asist event detected"
        elif "objective defense" in ocr_text.lower():
            text = "Objective defense event detected"
        elif "enemy trapped" in ocr_text.lower():
            text = "Enemy trapped event detected"
        else:
            text=""
            
        drawInfoBox(frame,p1,p2,text,ocr_text)    
        cv2.imshow('Frame',frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
            break

# When everything done, release the video capture object
stream.release()
 
# Closes all the frames
cv2.destroyAllWindows()

