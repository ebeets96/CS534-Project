import read_file as rf
import resize_image as ri
import os
import cv2

if __name__ == '__main__':
	directory = "flickr3"

	print("Reading file...")
	file_reader = rf.FileReader("many_images.txt")
	file_reader.next_images(directory, 50000)
	del file_reader

	print("Resizing images...")
	for filename in os.listdir(directory + '/'):
		img = cv2.imread(directory+'/'+filename, cv2.IMREAD_COLOR)
		try:
			img = ri.resize_image(img, 256, 256);
			cv2.imwrite(directory + '_cropped/'+filename, img)
		except:
			print("Skipping flickr/" + filename)
