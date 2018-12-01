import urllib
import shutil

# Moves all of the images in the text file to a folder called images

f = open("samplefile.txt", "r")
for line in f.readlines():
    url = line.split()[1]
    image = url.split("/")[4]
    urllib.urlretrieve(url, image)
    shutil.move(image, "images/" + image)

f.close()
