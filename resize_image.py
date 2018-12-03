import numpy as np
import cv2

def resize_image(img, w, h):
	height = img.shape[0]
	width = img.shape[1]

	if height < h or width < w:
		raise ValueError("Please enter an image with a height of at least 400 and a width of at least 400")

	scaled_img = img
	if (height - h) <= (width - w):
		new_width = int((h/height)*width)
		scaled_img = cv2.resize(img, (new_width, w))
		i = 0
		left = 0
		right = new_width
		total_height = new_width
		while (total_height != h):
			if i % 2 == 0:
				left += 1
			else:
				right -= 1
			total_height = right - left
		crop_img = scaled_img[0:w, left:right]
	else:
		new_height = int(w/width)*height
		scaled_img = cv2.resize(img, (h, new_height))
		i = 0
		top = 0
		bottom = new_height
		total_height = new_height
		while (total_height != h):
			if i % 2 == 0:
				top += 1
			else:
				bottom -= 1
			total_height = bottom - top

		crop_img = scaled_img[top:bottom, 0:w]

#	cv2.imshow("Display Window", scaled_img)
#	cv2.imshow("Display Window", crop_img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
	return crop_img

#m = resize_image(cv2.imread("cow.jpg"), 400, 400);
