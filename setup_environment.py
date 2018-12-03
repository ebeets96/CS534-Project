import read_file as rf
import resize_image as ri
import os
import cv2

file_reader = rf.FileReader("img_urls.txt")
file_reader.next_images("train", 1000)
file_reader.next_images("test", 200)
del file_reader


for filename in os.listdir('train/'):
	img = cv2.imread('train/'+filename, cv2.IMREAD_COLOR)
	try:
		img = ri.resize_image(img, 256, 256);
		cv2.imwrite('train/'+filename, img)
	except:
		print("Removing train/" + filename)
		os.remove('train/'+filename)
