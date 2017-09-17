import csv
import numpy as np
import urllib3
import urllib
import cv2
from urllib import request
from cv2 import __version__
__version__
print(__version__)
 
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    #resp1 = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image

with open('outside_front.csv') as f:
	readCSV = csv.reader(f, delimiter=',')
	i =0
	for row in readCSV:
		# Try 20 samples
		if i<=2:
			label = row[0]
			imageURL = row[1]
			i = i+1
			print("i", i)
			#print(imageURL)
			print ("downloading %s" % (imageURL))
			image = url_to_image(imageURL)
			
			print ("resizing images")
			resized_image = cv2.resize(image, (400, 400)) # resize image to 400*400
			###########################################################################################
			edged = cv2.Canny(resized_image, 10, 250)
			cv2.imshow("Edges", edged)
			cv2.waitKey(0)
			 
			#applying closing function 
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
			closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
			cv2.imshow("Closed", closed)
			cv2.waitKey(0)
			 
			#finding_contours 
			image, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			 
			# for c in cnts:
			# 	peri = cv2.arcLength(c, True)
			# 	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			# 	cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
			# cv2.imshow("Output", image)
			# cv2.waitKey(0)
			###########################################################################################
			idx = 0
			for c in contours:
				x,y,w,h = cv2.boundingRect(c)
				if w>50 and h>50:
					idx+=1
					new_img=image[y:y+h,x:x+w]
					cv2.imwrite(str(idx) + '.png', new_img)
			cv2.imshow("im",image)
			cv2.waitKey(0)
			###########################################################################################
			# get name from url
			name = imageURL.split('/').pop()
			print("Getting name of the image :", name)

			# show resized image in window
			#print("show resized image in window")
			#cv2.imshow(name, resized_image)
			#cv2.waitKey(0)
			
			# Save image
			print("Save Image")
			imageName = name + '.png'
			cv2.imwrite(imageName,resized_image) # save image 
			
			cv2.destroyAllWindows()
		
			# numpy array of image
			array = np.asarray(resized_image)
			print("Numpy array of images",array.shape)
			#print("RGB matrix: ", array)		# image RGB matrix


## LINKS
# https://github.com/andrewssobral/simple_vehicle_counting
# https://github.com/andrewssobral/simple_vehicle_counting/tree/master/python
# https://github.com/ksakmann/CarND-Vehicle-Detection
# https://arxiv.org/abs/1506.02640
# https://github.com/jeremy-shannon/CarND-Vehicle-Detection
# https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906
# http://mark-kay.net/2014/06/24/detecting-vehicles-cctv-image/
# https://github.com/thomasantony/CarND-P05-Vehicle-Detection
# https://github.com/udacity/CarND-Vehicle-Detection
