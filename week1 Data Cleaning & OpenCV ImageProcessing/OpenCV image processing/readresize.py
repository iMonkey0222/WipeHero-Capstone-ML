import numpy as np
import urllib
import cv2
 
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image


# initialize the list of image URLs to download
urls = [
    'https://tookan.s3.amazonaws.com/task_images/G0gC1486365254968-TOOKAN06022017061412.jpg',
    'https://tookan.s3.amazonaws.com/task_images/rQZD1486365263088-TOOKAN06022017061420.jpg',
    'https://tookan.s3.amazonaws.com/task_images/B7JB1486365339438-TOOKAN06022017061537.jpg',
]


# for url in urls:
#     # download the image URL and display it
#     print "downloading %s" % (url)
#     image = url_to_image(url)
#     cv2.imshow("Image", image)
#     k = cv2.waitKey(0) & 0xff
#     if k == 27:                 #  wiat for ESC key to quit
#         cv2.destroyAllWindows()
#     elif k == ord("s"):         # wait for 's' key to save and exit
#         cv2.imwrite("image.png",image)
#         cv2.destroyAllWindows()


for url in urls:
    # download the image URL and display it
    print "downloading %s" % (url)
    image = url_to_image(url)
    print "resizing images"
    resized_image = cv2.resize(image, (300, 300)) 
    cv2.imshow("resizedImage", resized_image)
    array = np.asarray(resized_image)
    print(array.shape)
    print("RGB matrix: ", array)
    k = cv2.waitKey(0) & 0xff
    if k == 27:                 #  wiat for ESC key to quit
        cv2.destroyAllWindows()
    elif k == ord("s"):         # wait for 's' key to save and exit
        cv2.imwrite("image.png",resized_image)
        cv2.destroyAllWindows()