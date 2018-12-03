import read_file as rf
import resize_image as ri

file_reader = rf.FileReader("img_urls.txt")
file_reader.next_images("train", 10)
#file_reader.next_images("test", 200)
del file_reader


for filename in os.listdir('train/'):
	img = cv2.imread('train/'+filename, cv2.IMREAD_COLOR)
	img = ri.resize_image(img, 256, 256);
	cv2.imwrite('train/'+filename, img)
