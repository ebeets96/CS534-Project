from flickrapi import FlickrAPI
from pprint import pprint
import json

FLICKR_PUBLIC = '92df2393ca8845a3dd398a795544c977'
FLICKR_SECRET = '91ae557f5a7409a5'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_l'
start_page = 1
number_per_page = 500
number_of_calls = 10

f = open("flickr_images2.txt","w+")

for i in range(start_page, number_of_calls + start_page):
	print(i)
	search = flickr.photos.search(text='landscape', per_page=number_per_page, page=i, extras=extras, safe_search='1', color_codes='5,6,7,8')
	photos = search['photos']
	for photo in photos['photo']:
		#parsed_json = json.loads(photo)
		try:
			print(photo["url_l"])
			f.write(photo['url_l'] + "\n")
		except:
			print("Skipping photo")
f.close()
