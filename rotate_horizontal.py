import cv2
import imutils


def rotate_horizontal_and_resize(im, size):
    if len(im) > len(im[0]):
        rotated_h = imutils.rotate_bound(im, 90)
    else:
        rotated_h = im
    resized_image = cv2.resize(rotated_h, (75, 30))
    return resized_image
