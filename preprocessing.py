import cv2
import numpy as np
import glob
import os
import pandas as pd
import filtering
import denoising
import masking


def subtract_roi(img, image_name, csv):
    image_id_new = int(image_name)
    selection = csv[csv.image_id == image_id_new]
    objects_vehicles = []
    for idx, row in selection.iterrows():
        point1 = [row.p1_x, row.p1_y]
        point2 = [row.p2_x, row.p2_y]
        point3 = [row.p3_x, row.p3_y]
        point4 = [row.p4_x, row.p4_y]
        objects_points = [[point1, point2, point3, point4]]
        objects_vehicles += objects_points
    mask = np.zeros(img.shape)
    for x in objects_vehicles:
        vrx = np.array(x, np.int32)
        vrx = vrx.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [vrx], (255, 255, 255))
    masked = np.where(mask < 1, img, np.zeros(img.shape))
    return masked


def preprocessing(im, img_name, df):
    denoised = denoising.denoise(im)
    bright = filtering.gamma_correction(denoised)
    contrast = filtering.sigmoid_correction(bright)
    histogram = filtering.clahehist(contrast)
    # subtracted = subtract_roi(im, img_name, df)
    mask1 = masking.create_mask(histogram, img_name, df)
    # dst = cv2.addWeighted(subtracted, 1, mask1, 1, 0)
    return mask1
