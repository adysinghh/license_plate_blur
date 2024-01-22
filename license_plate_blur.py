import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/car_plate.jpg')

#CREATE A FUNCTION THAT DISPLAYS THE IMAGE IN A LARGER SCALE AND CORRECT COLORING FOR MATPLOTLIB
def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

#LOAD THE HAARCASCADE_RUSSIAN_PLATE_NUMBER
plate_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_russian_plate_number.xml')

#CREATE A FUNCTION THAT TAKES IN AN IMAGE AND DRAWS A RECTANGLE AROUND WHAT IT DETECTS TO BE LICENSE PLATE 
#JUST DRAWING THE RECTABGLE AROUND THE PLATE
def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)

    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),4)

    return plate_img


## BLURING THE LICENSE PLATE
## GETING THE LICENSE PLATE IN ROI

def detect_and_blur_plate(img):
    plate_img = img.copy()
    roi = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)

    for (x,y,w,h) in plate_rects:
        roi = roi[y:y+h,x:x+w]     #y all the way to y+h then slice form x all the way to x+w
        blurred_roi = cv2.medianBlur(roi,7) #Blurring 
        plate_img[y:y+h,x:x+w] = blurred_roi
    return plate_img

result = detect_and_blur_plate(img)
display(result)
