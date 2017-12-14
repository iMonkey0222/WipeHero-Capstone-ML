import sys
import csv
import numpy as np 
import urllib
import cv2
from urllib import request
from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

def csv_image_reader(url, flag):
	#downloading images from csv file
	img= urllib.request.urlopen(url)
	image = np.asarray(bytearray(img.read()), dtype = "uint8")
	image = cv2.imdecode(image, flag)
	#image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def local_image_reader(url, flag):
	#downloading images from csv file
	img= urllib.request.urlopen('file://'+url)
	image = np.asarray(bytearray(img.read()), dtype = "uint8")
	image = cv2.imdecode(image, flag)
	#image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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

	#kernel = np.ones((5,5),np.float32)/25
	if switch == "avg":
		dst = cv2.blur(tmp,(5,5))
	elif switch == "gau":
		dst = cv2.GaussianBlur(tmp,(5,5),0)
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

	circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,75,
	                            param1=50,param2=25,minRadius=12,maxRadius=0)

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('detected circles',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def hough_ellipses(image):
	
	edges = canny(image, sigma=2.0,low_threshold=0.4, high_threshold=0.9)

	print("I am here now")
	# Perform a Hough Transform
	# The accuracy corresponds to the bin size of a major axis.
	# The value is chosen in order to get a single high accumulator.
	# The threshold eliminates low accumulators
	result = hough_ellipse(edges, accuracy=20, threshold=250)#, min_size=100, max_size=120)
	result.sort(order='accumulator')
	print("I am maybe here")

	# Estimated parameters for the ellipse
	best = list(result[-1])
	yc, xc, a, b = [int(round(x)) for x in best[1:5]]
	orientation = best[5]

	print("I am here now 2")
	# Draw the ellipse on the original image
	cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
	image_rgb[cy, cx] = (0, 0, 255)
	# Draw the edge (white) and the resulting ellipse (red)
	edges = color.gray2rgb(img_as_ubyte(edges))
	edges[cy, cx] = (250, 0, 0)

	fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True,subplot_kw={'adjustable':'box-forced'})
	ax1.set_title('Original picture')
	ax1.imshow(image_rgb)

	ax2.set_title('Edge (white) and result (red)')
	ax2.imshow(edges)

	plt.show()

def main():
	
	print ("Choose ",'\n', "1- local", '\n', "2- AWS ellipse" ,'\n', "3 - AWS Canny Hough ellipse", '\n', "4 AWS circles")
	choice = input()

	if choice == '1':
		with open('test.csv') as file:
			readCsv = csv.reader(file, delimiter = ',')
			
			for row in readCsv:
				label = row[0]
				url = row [1]
				print("Downloading image %a" % (url))
				image = local_image_reader(url, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR
				##PIPELINE				
				#Resizing
				image_resized = resize(image, 700, 700)
				
				#Resampling / Downsampling
				image_sampled = sampling(image_resized,0)
				
				#Smoothing
				image_smoothed_bil = smoothing(image_sampled,"bil")
				
				# Binarized w/ threshold
				image_binarized = threshold(image_smoothed_bil)

				# Contours		
				#image_contour = find_contours(image_binarized)
				#cv2.imshow(image_contour)

				#image_circles_contour = hough_circles(image_contour)
				image_circles = hough_circles(image_binarized)
				print('Trying to do ellipses')
				#hough_ellipses(image_contour)
				hough_ellipses(image_binarized)

	elif choice =='2':
		with open('outside_right.csv') as file:
			readCsv = csv.reader(file, delimiter = ',')
			
			for row in readCsv:
				label = row[0]
				url = row [1]
				print("Downloading image %a" % (url))
				image = csv_image_reader(url, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR
			
				#Resizing
				image_resized = resize(image, 700, 700)
				
				#Resampling / Downsampling
				image_sampled = sampling(image_resized,0)
				
				#Smoothing
				image_smoothed_bil = smoothing(image_sampled,"bil")
				
				# Binarized w/ threshold
				image_binarized = threshold(image_smoothed_bil)

				# Contours		
				#image_contour = find_contours(image_binarized)
				#cv2.imshow(image_contour)

				#image_circles_contour = hough_circles(image_contour)
				image_circles = hough_circles(image_binarized)
				print('Trying to do ellipses')
				#hough_ellipses(image_contour)
				#hough_ellipses(image_binarized)

	elif choice == '3':
		with open('outside_left.csv') as file:
			readCsv = csv.reader(file, delimiter = ',')
			
			for row in readCsv:
				label = row[0]
				url = row [1]
				print("Downloading image %a" % (url))
				image = csv_image_reader(url, cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR
			
				#Resizing
				image_resized = resize(image, 700, 700)
				hough_ellipses(image_binarized)

	else:
		with open('outside_right.csv') as file:
			readCsv = csv.reader(file, delimiter = ',')
			
			for row in readCsv:
				label = row[0]
				url = row [1]
				print("Downloading image %a" % (url))
				image = csv_image_reader(url, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_COLOR

				##PIPELINE
				
				#Resizing
				image_resized = resize(image, 700, 700)
				
				#Resampling / Downsampling
				image_sampled = sampling(image_resized,0)
				
				#Smoothing
				image_smoothed_avg = smoothing(image_sampled,"avg") # avg, gau, med, bil
				image_smoothed_gau = smoothing(image_sampled,"gau") # avg, gau, med, bil
				image_smoothed_med = smoothing(image_sampled,"med") # avg, gau, med, bil
				image_smoothed_bil = smoothing(image_sampled,"bil")
				
				#Smoothing Graphs
				titles = ['Original Image','Image sampled B/W ','image_smoothed_avg', 'image_smoothed_gau', 'image_smoothed_med', 'image_smoothed_bil']
				images = [image_resized, image_sampled, image_smoothed_avg, image_smoothed_gau, image_smoothed_med, image_smoothed_bil]
				for i in range(6):
					plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
					plt.title(titles[i])
					plt.xticks([]),plt.yticks([])
				plt.show()

				# Binarized w/ threshold and adaptive threshold
				image_binarized_list = threshold(image_smoothed_bil)
				image_adaptive_thres_list = adaptive_threshold(image_smoothed_bil)
				
				#Contouring
				ids = 0
				image_contour_list = []
				for im in image_binarized_list:
					image_contour = find_contours(im)
					image_contour_list.append(image_contour)
					#cv2.imwrite('contour' +str(ids) + '.png', image_contour)
					ids+=1

				titles_c = ['Original Image','Image sampled B/W ','image_smoothed_avg', 'image_smoothed_gau', 'image_smoothed_med', 'image_smoothed_bil']
				for i in range(6):
					plt.subplot(2,3,i+1),plt.imshow(image_contour_list[i],'gray')
					plt.title(titles_c[i])
					plt.xticks([]),plt.yticks([])
				plt.show()

				# print(len(image_contour_list))
				# print(image_contour_list[0])
				# print("Proceeding to HoughCircles")
				# #Circle detection
				# ids = 0
				# image_circles_list = []
				# for im in image_contour_list:
				# 	image_circles = hough_circles(im)
				# 	image_circles_list.append(image_circles)
				# 	cv2.imwrite('circles' +str(ids) + '.png', image_circles)

				# image_circles = hough_circles(image_binarized_list[4])
				# image_circles = hough_circles(image_binarized_list[5])
				# image_circles_adp = hough_circles(image_adaptive_thres_list[2])
				image_circles_contour = hough_circles(image_contour_list[0])
				# image_circles_contour = hough_circles(image_contour_list[2])
				# image_circles_contour = hough_circles(image_contour_list[3])
				image_circles_contour = hough_circles(image_contour_list[4])
				image_circles_contour = hough_circles(image_contour_list[5])
				#hough_ellipses(image_resized)

if __name__ == "__main__": main()
