

# id is primaryImage

# fillDB.py
# creates and populates SQL db

import urllib, json
import requests
import pprint
# https://www.geeksforgeeks.org/python-pil-image-show-method/
from PIL import Image
import time

import PIL as pil
import numpy as np
import scipy
import scipy.cluster
import scipy.spatial.distance as dis
from io import BytesIO


import csv
import ast


from numpy import loadtxt



NUM_CLUSTERS = 5




## finds palette, returns rgb values in decreasing order of use
def find_palette(image, num_clust):
	im = image.resize((100, int((image.height * 100) / image.width) ))      # resize to reduce time
	ar = np.asarray(im)
	shape = ar.shape
	ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value
	ar = ar[:,:3] # accounts for PNG files, with 4th alpha balue

	codes, dist = scipy.cluster.vq.kmeans(ar, num_clust) # performs k-means on image
	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = np.histogram(vecs, len(codes))    # count occurrences

	n = len(counts)

	idx = (-counts).argsort()[:n]
	colors = np.zeros((5,3))
	allcolors = codes[idx]

	# edge cases, if k-means returns fewer than 5 clusters
	if (n >= 5):
		colors = allcolors 
	else:
		for i in range(0, n):
		 	colors[i] = allcolors[i]

		if (n > 2):
			for j in range(n, 5):
				colors[j] = allcolors[j-n]
		else:
			for j in range(n,5):
				colors[j] = colors[0]
	return colors 



def main():

## NEW CSV FILE WITH ALL OPEN ACCESS MET OBJECTS
	# with open('MetObjects.txt', 'r') as inp, open('output.csv', 'w') as out:
	# 	writer = csv.writer(out)
	# 	for row in csv.reader(inp):
			
	# 	    if row[3] != "False":
	# 	    	writer.writerow(row)


## HOW TO READ THE COLORS AS FLOATS
	# reader = csv.reader(open('withimage.csv', 'r'))
	# rows = list(reader)
	# lis = rows[1][55]
	# tes = ast.literal_eval(lis)
	# print(type(tes[0][0])



# POPULATE DATABASE WITH COLORS
	start = time.time()


	reader = csv.reader(open('copyout.csv', 'r'))
	writer = csv.writer(open('final.csv', 'a'))
	# headers = next(reader)
	# headers.append("IMAGE URL")
	# headers.append("COLOR PALETTE")
	# writer.writerow(headers)

	rows = list(reader)

	TOTAL_ARTIFACTS = 239371

	# 238820, TOTAL_ARTIFACTS
	for i in range(238820, TOTAL_ARTIFACTS):
		id = rows[i][0]

		if i % 10 == 0:
			print("\n" + str(i))
			print("progress: " + str(int(100*i/TOTAL_ARTIFACTS)) + "%")
			print("time elapsed: " + str(int((time.time()-start)/60)) + " minutes")


		url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/" + str(id)
		r = requests.get(url)
		if r.status_code == 404:
			continue

		if r.status_code == 200:
			json = r.json()
			imageURL = json["primaryImage"]

			if imageURL == "": 
				continue

			response = requests.get(imageURL) # gets the image url

			if response.status_code == 404:
				continue


			rows[i].append(imageURL)

			ima = Image.open(BytesIO(response.content))
			ima = ima.convert('RGB')

			colors = find_palette(ima, NUM_CLUSTERS) 
			rows[i].append(colors.tolist())

			writer.writerow(rows[i])






	# for i in range(0, 1):
	#   try:

	#     # # print status
	#     # if i % 100 == 0:
	#     #     print("\n" + str(i))
	#     #     print("progress: " + str(int(100*i/TOTAL_ARTIFACTS)) + "%")
	#     #     print("time elapsed: " + str(int((time.time()-start)/60)) + " minutes")

	#     id = 11114

	#     url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/" + str(id)

	#     r = requests.get(url)

	#     if r.status_code == 404:
	#         continue

	#     if r.status_code == 200:
	#         json = r.json()


	#         imageURL = json["primaryImage"]


	#         if imageURL == "":
	#             continue

	#         print(imageURL)
	#         # response = requests.get(imageURL) # gets the image url
	#         # ima = Image.open(BytesIO(response.content))


	#         # colors = find_palette(ima, NUM_CLUSTERS) 
	#         # c1r = colors[0][0] 
	#         # c1g = colors[0][1] 
	#         # c1b = colors[0][2] 

	#         # c2r = colors[1][0] 
	#         # c2g = colors[1][1] 
	#         # c2b = colors[1][2]
	        
	#         # c3r = colors[2][0] 
	#         # c3g = colors[2][1] 
	#         # c3b = colors[2][2] 
	        
	#         # c4r = colors[3][0] 
	#         # c4g = colors[3][1] 
	#         # c4b = colors[3][2]
	        
	#         # c5r = colors[4][0] 
	#         # c5g = colors[4][1] 
	#         # c5b = colors[4][2]

	#   except:
	#     pass


main()
