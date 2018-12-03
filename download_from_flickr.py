from flickrapi import FlickrAPI
import json

FLICKR_PUBLIC = '92df2393ca8845a3dd398a795544c977'
FLICKR_SECRET = '91ae557f5a7409a5'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_n'
number_of_images = 50
number_per_page = 50
number_of_calls = round(number_of_images/number_per_page)

f = open("flickr_images.txt","w+")

for i in range(0, number_of_calls):
	print(i)
	search = flickr.photos.search(text='landscape', per_page=number_per_page, extras=extras, safe_search='1', color_codes='5,6,7,8')
	photos = search['photos']
	from pprint import pprint
	for photo in photos['photo']:
		#parsed_json = json.loads(photo)
		print(photo['url_n'] + "\n")
		f.write(photo['url_n'] + "\n")

f.close()
