import read_file as rf

file_reader = rf.FileReader("img_urls.txt")

while file_reader.next_images("foldername", 10):
	user_input = input("type in end to end loop\n")
	if(user_input == "end"):
		break

del file_reader
