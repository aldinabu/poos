import numpy as np
import cv2
import pandas as pd
import glob
import os


def create_mask_of_tag(img, image_name, csv):
    dst_dir = 'data2/masked_val/'
    image_id_new = int(image_name)
    selection = csv[csv.image_id == image_id_new]

    for idx, row in selection.iterrows():
        point1 = [row.p1_x, row.p1_y]
        point2 = [row.p2_x, row.p2_y]
        point3 = [row.p3_x, row.p3_y]
        point4 = [row.p4_x, row.p4_y]
        objects_points = [[point1, point2, point3, point4]]
        mask = np.zeros(img.shape)
        vrx = np.array(objects_points, np.int32)
        vrx = vrx.reshape((-1, 1, 2))
        mask = cv2.fillPoly(mask, [vrx], (255, 255, 255))
        masked = np.where(mask > 1, img, np.zeros(img.shape))
        if str(row.general_class) == "small vehicle":
            add_dst = 'small vehicle/'
            if str(row.sub_class) == "sedan":
                cv2.imwrite(dst_dir + add_dst + 'sedan/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "hatchback":
                cv2.imwrite(dst_dir + add_dst + 'hatchback/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "minivan":
                cv2.imwrite(dst_dir + add_dst + 'minivan/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "van":
                cv2.imwrite(dst_dir + add_dst + 'van/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "jeep":
                cv2.imwrite(dst_dir + add_dst + 'jeep/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "pickup":
                cv2.imwrite(dst_dir + add_dst + 'pickup/' + str(row.tag_id) + '.jpg', masked)
        else:
            add_dst = 'large vehicle/'
            if str(row.sub_class) == "dedicated agricultural vehicle":
                cv2.imwrite(dst_dir + add_dst + 'dedicated agricultural vehicle/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "prime mover":
                cv2.imwrite(dst_dir + add_dst + 'prime mover/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "truck":
                cv2.imwrite(dst_dir + add_dst + 'truck/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "bus":
                cv2.imwrite(dst_dir + add_dst + 'bus/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "minibus":
                cv2.imwrite(dst_dir + add_dst + 'minibus/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "light truck":
                cv2.imwrite(dst_dir + add_dst + 'light truck/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "crane truck":
                cv2.imwrite(dst_dir + add_dst + 'crane truck/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "cement mixer":
                cv2.imwrite(dst_dir + add_dst + 'cement mixer/' + str(row.tag_id) + '.jpg', masked)
            elif str(row.sub_class) == "tanker":
                cv2.imwrite(dst_dir + add_dst + 'tanker/' + str(row.tag_id) + '.jpg', masked)
