import h5py
import numpy as np
import cv2
import urllib
import csv

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # return the image
    return image

with open('outside_front.csv') as f:
    readCSV = csv.reader(f, delimiter=',')
    interestingrows=[row for idx, row in enumerate(readCSV) if idx in list(range(0,12329))]
#     interestingrows=[row for idx, row in enumerate(readCSV) if idx in list(range(0,5))]
    
    a = np.array([]).reshape(0,256,256,3)
    for row in interestingrows:
        label = row[0]
        imageURL = row[1]
        print ('downloading image %s'%(a.shape[0]))
        image = url_to_image(imageURL)
        
        resized_image = cv2.resize(image, (256, 256)) # resize image to 256*256
        
        # reshape numpy array
        reshape_img = resized_image.reshape((1,256,256,3))
        a = np.append(a,reshape_img,axis=0)
        
#         np.savetxt('test.txt',reshape_img,delimiter=',')
        
    
    # save as hdf5 file
    f = h5py.File('out_front_h5_local.h5','w')
    f['data']=a
    f.close()