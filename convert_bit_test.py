import numpy as np
import cv2
import convert_bit as cb

img = cv2.imread('images/hello.png', cv2.IMREAD_COLOR)
eight_bit = cb.convert_bit(3, img);
img2 = cb.convert_bit(8, eight_bit, 3);

cv2.imshow('original',img)
cv2.imshow('3-bit',eight_bit)
cv2.imshow('3-bit converted to 8-bit',img2)
cv2.waitKey(0)
