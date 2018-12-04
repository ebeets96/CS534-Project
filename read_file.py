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
		# shutil.rmtree(foldername, True)
		# os.makedirs(foldername)


		# loop through the next set of images and download them to the folder
		i = 1
		while(i <= number_of_images):
			print("Downloading Image %d", end="\r")
			url = self.file.readline().rstrip("\r\n")
			if(url == ''):
				return False

			#url = line.split()[1]
			image = url.split("/")[-1]

			try:
				download = requests.get(url, allow_redirects=False, timeout=10)
				if download.status_code == 200:
					with open(foldername + "/" + image, 'wb') as f:
						f.write(download.content)
				else:
					raise Exception("Status code was %d rather than 200" % download.status_code)

				i = i + 1
			except:
				# ignore the current readline
				print("Could not open: " + url);

		return True
