import read_file as rf
import resize_image as ri
import os
import cv2

# file_reader = rf.FileReader("img_urls.txt")
# file_reader.next_images("train", 1000)
# file_reader.next_images("test", 200)
# del file_reader


for filename in os.listdir('test/'):
	img = cv2.imread('test/'+filename, cv2.IMREAD_COLOR)
	try:
		img = ri.resize_image(img, 256, 256);
		cv2.imwrite('test_cropped/'+filename, img)
	except:
		print("Skipping test/" + filename)
