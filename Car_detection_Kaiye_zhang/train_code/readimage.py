import csv
import numpy as np
import urllib.request
import cv2
import os

if not os.path.exists('pos'):
    os.makedirs('pos')

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


with open('outside_front.csv') as f:
    readCSV = csv.reader(f, delimiter=',')

    i = 0
    img_index=1
    for row in readCSV:
        # Try 20 samples
            label = row[0]
            imageURL = row[1]
            # print(imageURL)
            print
            "downloading %s" % (imageURL)
            try:
                 print(imageURL)
                 urllib.request.urlretrieve(imageURL, 'neg/' + str(img_index) + '.jpg')
                   # 把图片转为灰度图片
                 gray_img = cv2.imread('neg/' + str(img_index) + '.jpg', cv2.IMREAD_GRAYSCALE)
                   # 更改图像大小
                 image = cv2.resize(gray_img, (100, 100))
                 # 保存图片
                 cv2.imwrite('neg/' + str(img_index) + '.jpg', image)
                 img_index += 1
            except Exception as e:
                 print(e)

        # image = url_to_image(imageURL)
        #
        #     print
        #     "resizing images"
        #     resized_image = cv2.resize(image, (500, 500))  # resize image to 400*400
        #
        #     # get name from url
        #     name = imageURL.split('/').pop()
        #
        #     # show resized image in window
        #     cv2.imshow(name, resized_image)
        #     cv2.waitKey(0)
        #
        #     # Save image
        #     imageName = name + '.png'
        #     cv2.imwrite(imageName, resized_image)  # save image
        #
        #     cv2.destroyAllWindows()
        #
        #     # numpy array of image
        #     array = np.asarray(resized_image)
        #     print(array.shape)
        #     # print("RGB matrix: ", array)		# image RGB matrix
        #
        #
        #
        #     # k = cv2.waitKey(0) & 0xff
        #     # if k == 27:                 #  wiat for ESC key to quit
        #     #     cv2.destroyAllWindows()
        #     # elif k == ord("s"):         # wait for 's' key to save and exit
        #     #     cv2.imwrite("image.png",resized_image)
        #     #     cv2.destroyAllWindows()
