import cv2
import numpy
import Tkinter
import matplotlib.pyplot as plt

def BW(imageName):
    img = cv2.imread(imageName)
    imggrey = cv2.imread(imageName, 0)
    cv2.imwrite(img + '.png', imggrey)
