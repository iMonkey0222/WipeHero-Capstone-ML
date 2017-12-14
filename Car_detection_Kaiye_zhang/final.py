import numpy as np
import cv2
import argparse
import os
import datetime
import time
import urllib.request
import sys
from PIL import Image

_data_path = 'data'
# this is the cascade we just made. Call what you want
cars_cascade = cv2.CascadeClassifier('all.xml')

def _gen_timestamp():
    """
    Generate a random prefix for image file
    :return: string prefix
    """
    timestamp = str(int(time.time() * 1000))[-5:-1]
    prefix = timestamp
    return prefix

def detected(image):
    img =cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = cars_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300), maxSize=(900, 900))
    if len(cars) ==1:
        return True
    else:
        return False

def show(image,img_type='jpg'):
    if not os.path.exists(_data_path):
        os.mkdir(_data_path)

    try:
        # Fetch image stream from response
        img = cv2.imread(image)
    except Exception as err:
        raise Exception('%s opening image file failed: %s' % (image, err))

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300),
                                             maxSize=(900, 900))
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        img_new =img
    except Exception as err:
        raise Exception('Sizing image file: %s error: %s' % (image, err))

    # Generate a random prefix for image file
    img_prefix = _gen_timestamp()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(_data_path, dir_prefix.strftime('%Y%m%d'))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subdir_path = os.path.join(dir_path, dir_prefix.strftime('%H'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    local_path = os.path.join(subdir_path, img_localname)
    if os.path.exists(local_path):
        img_prefix = _gen_timestamp()
        img_localname = '%s.%s' % (img_prefix, img_type)
        local_path = os.path.join(subdir_path, img_localname)
        if os.path.exists(local_path):
            raise Exception('%s image name existed' % local_path)

    try:
        # Save the image
        try:
            im = Image.fromarray(img_new)
            im.save(local_path)
        except IOError:
            im.convert('RGB').save(local_path)
    except Exception as err:
        raise Exception('Image file saving failed: %s' % err)

    return local_path



def crop_image_url(url, img_type='jpg', quality=100):
    """
    :param url: source image url
    :param width: cropped width
    :param height: cropped height
    :param img_type: cropped image type
    :param quality: cropped image quality
    """
    if not os.path.exists(_data_path):
        os.mkdir(_data_path)
    try:
        urllib.request.urlretrieve(url, 'data/' + 'sample' + '.jpg')
    except Exception as err:
        raise Exception('%s request error: %s' % (url, str(err)))

    try:
        # Fetch image stream from response
        img = cv2.imread('data/' + 'sample' + '.jpg')
    except Exception as err:
        raise Exception('%s opening image failed: %s' % (url, err))

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300),
                                             maxSize=(900, 900))
        for (x, y, w, h) in cars:
            img_new = img.crop((x, y, w, h))
    except Exception as err:
        raise Exception('detection failed: %s error: %s' % (url, err))

    # Generate a random prefix for image file
    img_prefix = _gen_timestamp()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(_data_path, dir_prefix.strftime('%Y%m%d'))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subdir_path = os.path.join(dir_path, dir_prefix.strftime('%H'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    local_path = os.path.join(subdir_path, img_localname)
    if os.path.exists(local_path):
        img_prefix = _gen_timestamp()
        img_localname = '%s.%s' % (img_prefix, img_type)
        local_path = os.path.join(subdir_path, img_localname)
        if os.path.exists(local_path):
            raise Exception('%s image name existed' % local_path)

    try:
        # Save the image
        try:
            im = Image.fromarray(img_new)
            im.save(local_path)
        except IOError:
            im.convert('RGB').save(local_path)
    except Exception as err:
        raise Exception('Image file saving failed: %s' % err)

    return local_path
def crop_image_file(filename, img_type='jpg', quality=100):
    """
    :param url: source image base64 encoded stream
    :param width: cropped width
    :param height: cropped height
    :param img_type: cropped image type
    :param quality: cropped image quality
    """
    if not os.path.exists(_data_path):
        os.mkdir(_data_path)

    try:
        # Fetch image stream from response
        img = cv2.imread(filename)
    except Exception as err:
        raise Exception('%s opening image file failed: %s' % (filename, err))

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(300, 300),
                                             maxSize=(900, 900))
        for (x, y, w, h) in cars:
            img_new = img.crop((x, y, w, h))

    except Exception as err:
        raise Exception('Sizing image file: %s error: %s' % (filename, err))

    # Generate a random prefix for image file
    img_prefix = _gen_timestamp()
    img_localname = '%s.%s' % (img_prefix, img_type)

    # Put it into directory by the sequence of date(yyyymmdd/h)
    dir_prefix = datetime.datetime.now()
    dir_path = os.path.join(_data_path, dir_prefix.strftime('%Y%m%d'))
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    subdir_path = os.path.join(dir_path, dir_prefix.strftime('%H'))
    if not os.path.exists(subdir_path):
        os.mkdir(subdir_path)

    local_path = os.path.join(subdir_path, img_localname)
    if os.path.exists(local_path):
        img_prefix = _gen_timestamp()
        img_localname = '%s.%s' % (img_prefix, img_type)
        local_path = os.path.join(subdir_path, img_localname)
        if os.path.exists(local_path):
            raise Exception('%s image name existed' % local_path)

    try:
        # Save the image
        try:
            im = Image.fromarray(img_new)
            im.save(local_path)
        except IOError:
            im.convert('RGB').save(local_path)
    except Exception as err:
        raise Exception('Image file saving failed: %s' % err)

    return local_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Car detection and cropper')
    result = {}
    try:
        parser.add_argument('-i', action='store', required=True,
                            dest='input',
                            help='Url or filename of source image')
        parser.add_argument('-q', action='store', default=100, type=int,
                            dest='quality',
                            help='Quality of cropped image, 1 to 100')
        parser.add_argument('-t', action='store', default='jpg',
                            dest='type',
                            help='File type of cropped image, default is jpg')
        parser.add_argument('-m', action='store', default='show',
                            dest='mode',
                            help='Imaging mode: url | file |show')
        result = parser.parse_args(sys.argv[1:])
    except Exception as err:
        print('Params error: %s' % err)
        exit(1)

    try:
        mode = result.mode
        source = result.input
        if mode == 'file':
            local_path = crop_image_file(source,
                                         img_type=result.type,
                                         quality=result.quality)
            print('Cropped image file successfully! Image path is %s' % local_path)
        elif mode == 'url':
            local_path = crop_image_url(source,
                                        img_type=result.type,
                                        quality=result.quality)
            print('Cropped remote image successfully! Image path is %s' % local_path)

        elif mode == 'show':
            print('here')
            local_path = show(source, img_type=result.type)
            print('Image is here! %s' % local_path)
        else:
            raise Exception('Unhandled args')
    except Exception as err:
        print(err)