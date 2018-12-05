import requests
import shutil
import os
import multiprocessing as mp

# Moves all of the images in the text file to a folder called images
def download_file (url, foldername):
	image = url.split("/")[-1]
	if(os.path.isfile(image)):
		print("Skipping " + image)
	 	return

	download = requests.get(url, allow_redirects=False, timeout=10)
	try:
		if download.status_code == 200:
			with open(foldername + "/" + image, 'wb') as f:
				f.write(download.content)
		else:
			print("Status code was {0} rather than 200".format(download.status_code))
	except Exception as err:
		# ignore the current readline
		print("Could not open: " + url)
		print("\t" + err)

class FileReader:
	def __init__(self, filename):
		self.file = open(filename, "r");

	def __del__(self):
		self.file.close();

	def hello():
		return 1

	# returns false if EOF is reached
	def next_images(self, foldername, number_of_images):
		# delete folder and all images and recreate it
		# shutil.rmtree(foldername, True)
		# os.makedirs(foldername)

		pool = mp.Pool(10) # 10 images can download at the same time

		# loop through the next set of images and download them to the folder
		for i in range(1, number_of_images + 1):
			#print("Downloading Image {0}".format(i), end="\r")
			url = self.file.readline().rstrip("\r\n")
			if(url == ''):
				pool.close()
				pool.join()
				return False

			# self.download_file(url, foldername)
			# resuuu = pool.apply_async(self.hello)
			res = pool.apply_async(download_file, args=(url, foldername,))

		pool.close()
		pool.join()

		return True
