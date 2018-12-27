import numpy as np
import cv2
import pandas as pd
import glob
import os


def label_vehicle(img, image_name, csv, dict):
    image_id_new = int(image_name)
    selection = csv[csv.image_id == image_id_new]
    windowed_with_name = img
    labels_index = {0: "hatchback", 1: "jeep", 2: "minivan", 3: "pickup", 4: "sedan", 5: "van"}
    for idx, row in selection.iterrows():
        p1 = [row.p1_x, row.p1_y]
        p2 = [row.p2_x, row.p2_y]
        p3 = [row.p3_x, row.p3_y]
        p4 = [row.p4_x, row.p4_y]

        contour = [p1, p2, p3, p4]
        ctr = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
        windowed = img
        if str(row.general_class) == 'small vehicle':
            windowed = cv2.drawContours(img, [ctr], -1, (0, 255, 0), 2)

        miniX = min(row.p1_x, row.p2_x, row.p3_x, row.p4_x)
        maxiX = max(row.p1_x, row.p2_x, row.p3_x, row.p4_x)

        miniY = min(row.p1_y, row.p2_y, row.p3_y, row.p4_y)
        maxiY = max(row.p1_y, row.p2_y, row.p3_y, row.p4_y)

        x = round(miniX + (maxiX - miniX) / 2)
        y = round(miniY + (maxiY - miniY) / 2)
        for tag, value in dict.items():
            if tag == int(row.tag_id):
                windowed_with_name = cv2.putText(windowed, labels_index.get(value),
                                                 (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

    return windowed_with_name
