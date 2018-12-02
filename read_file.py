import requests
import shutil
import os

# Moves all of the images in the text file to a folder called images
class FileReader:
	def __init__(self, filename):
		self.file = open(filename, "r");

	def __del__(self):
		self.file.close();

	# returns false if EOF is reached
	def next_images(self, foldername, number_of_images):
		# delete folder and all images and recreate it
		shutil.rmtree(foldername, True)
		os.makedirs(foldername)


		# loop through the next set of images and download them to the folder
		i = 0
		while(i < number_of_images):
			line = self.file.readline()
			if(line == ''):
				return False

			url = line.split()[1]
			image = url.split("/")[-1]

			try:
				download = requests.get(url, allow_redirects=False, timeout=5)
				if download.status_code == 200:
					with open(foldername + "/" + image, 'wb') as f:
						f.write(download.content)
				else:
					raise Exception()

				i = i + 1
			except:
				# ignore the current readline
				print("Could not open: " + url);

		return True
