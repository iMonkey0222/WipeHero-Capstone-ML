import sys
import csv
import numpy as np 
import urllib
import cv2
from urllib import request
from matplotlib import pyplot as plt

def csv_image_reader(url, flag):
	#downloading images from csv file
	img= urllib.request.urlopen(url)
	image = np.asarray(bytearray(img.read()), dtype = "uint8")
	image = cv2.imdecode(image, flag)
#	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def sampling(image, switch):
	## Upsampling/downsampling using Gaussian and Laplacian pyramid techniques
	window_name = "Sampling window"
	#Create window
	cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
	tmp = image
	dst = tmp
	#cv2.imshow( window_name, dst)
	#cv2.waitKey(0)

	if switch == 1:
		dst = cv2.pyrUp(tmp)
	elif switch == 0:
		dst = cv2.pyrDown(tmp)

	cv2.imshow(window_name, dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	tmp = dst
	return tmp

def resize(image, row, col):
	resized_image = cv2.resize(image, (row, col))
	return resized_image

def smoothing(image, switch):
	window_name = "Smoothing window"
	cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
	tmp = image
	dst = tmp

#	kernel = np.ones((5,5),np.float32)/25
	if switch == "avg":
		dst = cv2.blur(tmp,(5,5))
	elif switch == "gau":
		dst = cv2.GaussianBlur(tmp,(5,5))
	elif switch == "med":
		dst = cv2.medianBlur(tmp, 5)
	elif switch == "bil":			###IMPORTANT AS IT KEEPS EDGES
		dst = cv2.bilateralFilter(tmp,5,45,45)

	cv2.imshow(window_name, dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	tmp = dst
	return tmp

def threshold(image):
	#window_name = "Smoothing window"
	#cv2.namedWindow( window_name, cv2.WINDOW_NORMAL)
	tmp = image
	dst = tmp
	print("cv2.THRESH_BINARY = ",cv2.THRESH_BINARY)
	ret,thresh1 = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(tmp,127,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(tmp,127,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(tmp,127,255,cv2.THRESH_TOZERO_INV)

	titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'FOR CONTOUR']
	images = [tmp, thresh1, thresh2, thresh3, thresh4, thresh5]
	for i in range(6):
	    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	plt.show()

	return images

def adaptive_threshold(image):
	tmp = image
	dst = tmp

	ret,th1 = cv2.threshold(tmp,127,255,cv2.THRESH_BINARY)
	th2 = cv2.adaptiveThreshold(tmp,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(tmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,11,2)
	titles = ['Original Image', 'Global Thresholding (v = 127)',
	            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [tmp, th1, th2, th3]
	for i in range(4):
	    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	plt.show()

	return images

def find_contours(image):
	#--- Find all the contours in the binary image ---
	_, contours,hierarchy = cv2.findContours(image,2,1)
	cnt = contours
	big_contour = []
	max = 0
	for i in cnt:
		area = cv2.contourArea(i) #--- find the contour having biggest area ---
		if(area > max):
			max = area
			big_contour = i 

	final = cv2.drawContours(image, big_contour, -1, (0,255,0), 3)
	cv2.imshow('final', final)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return final

def hough_circles(image):
	tmp = image
	dst = tmp

	cimg = cv2.cvtColor(tmp,cv2.COLOR_GRAY2BGR)

	circles = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT,1,20,
	                            param1=60,param2=30,minRadius=0,maxRadius=0)

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
	    # draw the outer circle
	    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('detected circles',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	with open('outside_left.csv') as file:
		readCsv = csv.reader(file, delimiter = ',')
		
		for row in readCsv:
			label = row[0]
			url = row [1]
			print("Downloading image %a" % (url))
			image = csv_image_reader(url, cv2.IMREAD_GRAYSCALE)

			##PIPELINE
			
			image_resized = resize(image, 700, 700)
			
			image_sampled = sampling(image_resized,0)
			
			image_smoothed = smoothing(image_sampled,"avg") # avg, gau, med, bil
			
			image_smoothed_bil = smoothing(image_sampled,"bil")
			
			image_binarized_list = threshold(image_smoothed_bil)
			
			image_adaptive_thres_list = adaptive_threshold(image_smoothed_bil)
			
			image_contour = find_contours(image_binarized_list[0])

			image_circles = hough_circles(image_binarized_list[4])
			image_circles_adp = hough_circles(image_adaptive_thres_list[2])
			image_circles_contour = hough_circles(image_contour)
			#print(image.size, image.shape)
			#print(image_resized.size, image_resized.shape)
			#print(image_sampled.size, image_sampled.shape)			


if __name__ == "__main__": main()
