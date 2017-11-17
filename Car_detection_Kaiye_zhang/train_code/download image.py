import urllib.request
import cv2
import os

# 创建图片保存目录
if not os.path.exists('neg'):
    os.makedirs('neg')

neg_img_url = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00463246']

urls = ''
for img_url in neg_img_url:
    urls += urllib.request.urlopen(img_url).read().decode()

img_index = 1700
for url in urls.split('\n'):
    try:
        print(url)
        urllib.request.urlretrieve(url, 'neg/' + str(img_index) + '.jpg')
        # 把图片转为灰度图片
        gray_img = cv2.imread('neg/' + str(img_index) + '.jpg', cv2.IMREAD_GRAYSCALE)
        # 更改图像大小
        image = cv2.resize(gray_img, (500, 500))
        # 保存图片
        cv2.imwrite('neg/' + str(img_index) + '.jpg', image)
        img_index += 1
    except Exception as e:
        print(e)


# 判断两张图片是否完全一样
'''def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False
'''
