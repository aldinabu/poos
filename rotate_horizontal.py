import numpy as np
import cv2
import pandas as pd
import imutils
import glob
import os


def rotate_horizontal_and_resize(im, size):
    if len(im) > len(im[0]):
        rotated_h = imutils.rotate_bound(im, 90)
    else:
        rotated_h = im
    resized_image = cv2.resize(rotated_h, size)
    return resized_image


print("Working on it, please have patience")
src_dir = 'data/masked_rotated_val/small vehicle/van'
br = 0
for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
    im = cv2.imread(filename)
    name = filename.replace(src_dir, '')
    img_name = name.replace('.jpg', '')
    img_name = img_name.replace('\\', '')
    rot = rotate_horizontal_and_resize(im, (75, 30))
    cv2.imwrite('data/masked_rotated_h_val/small vehicle/van/' + img_name + '.jpg', rot)
    br += 1
    print(br)
print("Done")
