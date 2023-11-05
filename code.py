import numpy as np
import cv2 as cv

img= cv.imread('input_image.jpeg',0)
dark_img=cv.convertScaleAbs(img, alpha=0.5, beta=-50)
blur_img=cv.medianBlur(dark_img,5)
final_img=cv.bitwise_not(cv.inRange(blur_img, 0, 70)) 
contours, hierarchies = cv.findContours(final_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
img=cv.cvtColor(img, cv.COLOR_GRAY2RGB);
M = cv.moments(contours[0])
for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(img, [i], -1, (0, 255, 0), 2)
        cv.circle(img, (cx, cy), 7, (0, 0, 255), -1)
print(f"x: {cx} y: {cy}")

filename = 'output_image.jpg'
cv.imwrite(filename, img)