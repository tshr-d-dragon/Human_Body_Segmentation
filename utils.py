
"""
Modes:
0: Picture in BG
1: Video in BG
2. Blurred Picture in BG
3: Blurred Video in BG
4: B/W Picture in BG
5: B/W Vide in BG
"""



""" Imports """

import numpy as np
import cv2



""" Global Variables """

global counter
counter = 0



""" Helper Functions """

def Substract_BG(frame, pred, mode):

    mask = cv2.bitwise_and(frame, np.dstack([pred, pred, pred]))
    output = cv2.bitwise_or(mask, np.zeros_like(frame, dtype = np.uint8))
    
    if mode == 0:
        new_bg = Add_BG_image(output)
    elif mode == 1:
        new_bg = Add_BG_video(output)
    elif mode == 2:
        new_bg = Add_BG_image_Blurred(output)
    elif mode == 3:
        new_bg = Add_BG_video_Blurred(output)
    elif mode == 4:
        new_bg = Add_BG_image_BnW(output)
    elif mode == 5:
        new_bg = Add_BG_video_BnW(output)
    else:
        print('Invalid Mode')
    
    return new_bg


def Add_BG_image(output):
    
    bg = cv2.imread('Human_Seg/bg.jpg')
    bg = cv2.resize(bg, (512, 512))
        
    output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
    output_inv = cv2.bitwise_not(output_binary)
    temp = cv2.bitwise_and(bg, output_inv)
    new_bg = cv2.bitwise_or(temp, output)
    
    return new_bg


def Add_BG_video(output):
    
    global counter
    
    video_file = r"Human_Seg\Underwater.mp4"
    cap = cv2.VideoCapture(video_file)
    ret = True
    
    while ret :
        frameId = int(round(cap.get(1)))    
        ret, bg = cap.read() 
        bg = cv2.resize(bg, (512, 512))
        if frameId == 100:
            counter = 0 
        elif frameId == counter:
            output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
            output_inv = cv2.bitwise_not(output_binary)
            temp = cv2.bitwise_and(bg, output_inv)
            new_bg = cv2.bitwise_or(temp, output)
            counter += 1
            break
        else:
            continue
    return new_bg
   
    
def Add_BG_image_Blurred(output):
    
    bg = cv2.imread('Human_Seg/bg.jpg')
    bg = cv2.resize(bg, (512, 512))
    bg = cv2.GaussianBlur(bg,(11, 11),cv2.BORDER_DEFAULT)
        
    output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
    output_inv = cv2.bitwise_not(output_binary)
    temp = cv2.bitwise_and(bg, output_inv)
    new_bg = cv2.bitwise_or(temp, output)

    return new_bg


def Add_BG_video_Blurred(output):
    
    global counter
    
    video_file = r"Human_Seg\Underwater.mp4"
    cap = cv2.VideoCapture(video_file)
    ret = True
    
    while ret :
        frameId = int(round(cap.get(1)))    
        ret, bg = cap.read() 
        bg = cv2.resize(bg, (512, 512))
        bg = cv2.GaussianBlur(bg,(11, 11),cv2.BORDER_DEFAULT)
        if frameId == 50:
            counter = 0 
        elif frameId == counter:
            output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
            output_inv = cv2.bitwise_not(output_binary)
            temp = cv2.bitwise_and(bg, output_inv)
            new_bg = cv2.bitwise_or(temp, output)
            counter += 1
            break
        else:
            continue
        
    return new_bg


def Add_BG_image_BnW(output):
    
    bg = cv2.imread('Human_Seg/bg.jpg', 0)
    bg = cv2.resize(bg, (512, 512))
        
    output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
    output_inv = cv2.bitwise_not(output_binary)
    temp = cv2.bitwise_and(np.dstack([bg, bg, bg]), output_inv)
    new_bg = cv2.bitwise_or(temp, output)

    return new_bg


def Add_BG_video_BnW(output):
    
    global counter
    
    video_file = r"Human_Seg\Underwater.mp4"
    cap = cv2.VideoCapture(video_file)
    ret = True
    
    while ret :
        frameId = int(round(cap.get(1)))    
        ret, bg = cap.read() 
        bg = cv2.resize(bg, (512, 512))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        if frameId == 100:
            counter = 0 
        elif frameId == counter:
            output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)[1]
            output_inv = cv2.bitwise_not(output_binary)
            temp = cv2.bitwise_and(np.dstack([bg, bg, bg]), output_inv)
            new_bg = cv2.bitwise_or(temp, output)
            counter += 1
            break
        else:
            continue
        
    return new_bg