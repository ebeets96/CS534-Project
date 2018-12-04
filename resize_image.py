import numpy as np
import cv2

def resize_image(img, w, h):
	height = img.shape[0]
	width = img.shape[1]

	if height < h or width < w:
		raise ValueError("Image is smaller than the resize input")

	scaled_img = img
	if (height - h) <= (width - w):
		#image is scaled so that height is = h
		new_width = int((h/height)*width)
		scaled_img = cv2.resize(img, (new_width, w))
		# i = 0
		left = int((new_width - w)/2)
		right = left + w
		crop_img = scaled_img[0:w, left:right]
	else:
		new_height = int(w/width*height)
		scaled_img = cv2.resize(img, (w, new_height))
		i = 0
		top = int((new_height - h)/2)
		bottom = top + h
		crop_img = scaled_img[top:bottom, 0:w]

#	cv2.imshow("Display Window", scaled_img)
#	cv2.imshow("Display Window", crop_img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
	return crop_img

#m = resize_image(cv2.imread("cow.jpg"), 400, 400);
