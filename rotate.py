import numpy as np
import cv2
import pandas as pd
import imutils
import glob
import os


def rotate(im, tag, df):
    image_id_new = int(tag)
    selection = df[df.tag_id == image_id_new]

    objects_points = np.array([])
    for idx, row in selection.iterrows():
        if str(row.tag_id) == tag:
            point1 = [int(row.p1_x), int(row.p1_y)]
            point2 = [int(row.p2_x), int(row.p2_y)]
            point3 = [int(row.p3_x), int(row.p3_y)]
            point4 = [int(row.p4_x), int(row.p4_y)]
            objects_points = np.array([point1, point2, point3, point4])

    rect = cv2.minAreaRect(objects_points)
    angle = rect[2]
    rotated = imutils.rotate_bound(im, -angle)

    image = rotated
    edged = cv2.Canny(image, 10, 250)
    cnt = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    croped = image[y:y + h, x:x + w]
    return croped
