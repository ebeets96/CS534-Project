from flickrapi import FlickrAPI
from pprint import pprint
import json
import time

FLICKR_PUBLIC = '92df2393ca8845a3dd398a795544c977'
FLICKR_SECRET = '91ae557f5a7409a5'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_l'
start_page = 1
number_per_page = 400
number_of_calls = 10

fname = "many_images.txt"
if(os.path.isfile(fname))
 	raise Exception("File already exists")
	
f = open("many_images.txt","w+")
ts = int(time.time())
seconds_in_30_days = 60 * 60 * 24 * 30
total_images = 0

for month in range(24):
	end_time = ts - seconds_in_30_days * month
	start_time = end_time - seconds_in_30_days
	print("Searching from {0} to {1}".format(start_time, end_time))
	for i in range(start_page, number_of_calls + start_page):
		search = flickr.photos.search(
			min_upload_date = start_time,
			max_upload_date = end_time,
			text='landscape',
			per_page=number_per_page,
			page=i,
			extras=extras,
			safe_search='1',
			color_codes='5,6,7,8'
		)

		photos = search['photos']
		for photo in photos['photo']:
			#parsed_json = json.loads(photo)
			try:
				f.write(photo['url_l'] + "\n")
				total_images = total_images + 1
				print(str(total_images) + " : " + photo["url_l"])
			except Exception as err:
				print("Skipping photo: ", err)

f.close()
