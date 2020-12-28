#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

from matplotlib import pyplot as plt
import os
import pandas as pd
import math
from scipy import misc,ndimage
import csv
from imutils import perspective
from scipy.ndimage import interpolation as inter
import glob
import numpy as np
import os
import cv2
import math
from scipy import misc,ndimage
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_file = []
detect_result = []
imagelist_jpg = (glob.glob(r'AI/*.jpg'))
for aa in imagelist_jpg:
    img = Image.open(aa)
    img.save(r'AI/png/05.tiff')
    img = cv2.imread('AI/png/05.tiff',cv2.IMREAD_COLOR)
    img = cv2.resize(img, (820,680))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray =  cv2.blur(gray, (3,3))#Blur to reduce noise
    #gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    plt.imshow(gray)
    plt.show()
    plt.imshow(edged)
    plt.show()
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    screenCnt = None
    # loop over our contours
    for c in cnts:
      # approximate the contour
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.018* peri, True)
  # if our approximated contour has four points, then
  # we can assume that we have found our screen
          if len(approx) == 4:
                screenCnt = approx
                break
    if screenCnt is None:
        detected = 0
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(img, [screenCnt], 0, (0, 255, 0), 0)
        print("contour detected : " ,detected)
# Masking the part other than the number plate
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,0,)
        new_image = cv2.bitwise_and(img,img,mask=mask)
        # Now crop
        (x, y) = np.where(mask == 255)
        a_x = []
        a_y = []
        for i in screenCnt:
            a_x.append(min(i)[0])
            a_y.append(min(i)[1])
        (topx, topy) = (np.min(x), np.min(y))
        #print ((topx,topy))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        #print ((bottomx,bottomy))
        Cropped = img[topx:bottomx+1, topy:bottomy+1] 
        plt.imshow(Cropped)
        plt.show()
        pts = np.array([ (a_x[2]-1, a_y[2]),(a_x[3]-1,a_y[3]),(a_x[0]+1, a_y[0]), (a_x[0]+1,a_y[1])])
        warped = perspective.four_point_transform(img, pts)
        text = pytesseract.image_to_string(warped,lang = 'eng' ,config='--psm 11')
        #print (text)
        if text == "":
            pts = np.array([ (a_x[3], a_y[3]),(a_x[2],a_y[2]),(a_x[1], a_y[1]+1), (a_x[0],a_y[0]+1)])
            #plt.imsave('AI/png/05.png', Cropped)
    # font 
            warped = perspective.four_point_transform(img, pts)
            warped = cv2.resize(warped,(650,300))
    #font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    #org = (50, 50)   
    # fontScale n
    #fontScale = 1   
    # Blue color in BGR 
    #color = (255, 0, 0)   
    # Line thickness of 2 px 
    #thickness = 2
            text = pytesseract.image_to_string(warped,lang = 'eng' ,config='--psm 11')
    if detected == 0 or len(text) < 5 or (text).isspace() == True :
        #pts = np.array([ (200, 50),(200,365),(500, 0), (500,365)])
        #warped = perspective.four_point_transform(img, pts)
        
        print ('another')
        #print("No contour detected") 
        img = cv2.resize(img, (700,400) )
        warped = img
        text = pytesseract.image_to_string(warped,lang = 'eng' ,config='--psm 6')
        
    plt.imshow(warped)
    plt.show()
    #print (len(text))
    print ('text: ' + text)
    image_file.append(aa)
    detect_result.append(text)
image_file = pd.DataFrame(image_file, columns = {'image_file'})
detect_result = pd.DataFrame(detect_result, columns = {'detect_result'})
final  = pd.concat([image_file,detect_result],axis = 1)
final.to_csv('license_plate_test.csv',index = None)


# In[ ]:




