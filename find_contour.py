import numpy as np
import cv2 as cv
im = cv.imread('test.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
filename = 'gray_scale.jpg'
cv.imwrite(filename, thresh)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(im, contours, -1, (0,255,0), 3)
# filename = 'savedImage.jpg'
# cv.imwrite(filename, im)

cnt = contours[0]
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
filename = 'savedImage.jpg'
cv.imwrite(filename, im)