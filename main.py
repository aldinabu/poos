from tensorflow import python as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from random import shuffle
from label_vehicles import label_vehicle
import pandas as pd
import glob
import cv2
from preprocessing import preprocessing
from rotate import rotate
from rotate_horizontal import rotate_horizontal_and_resize
from preparation import create_mask_of_tag
from load_model import pretty_cm, transform_image, evaluation_indices, report


def main(path):
    print("Working on it, please have patience")
    src_dir = 'data2/' + path
    df = pd.read_csv('data2/val_csv.csv', sep=',')
    br = 0

    for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
        im = cv2.imread(filename)
        name = filename.replace(src_dir, '')
        img_name = name.replace('.jpg', '')
        img_name = img_name.replace('\\', '')
        preprocessed = preprocessing(im, img_name, df)
        create_mask_of_tag(preprocessed, img_name, df)
        br += 1
        print(br)
    src_dir = 'data2/masked_val/small vehicle'
    dst_dir = 'data2/masked_rotated_h_val/'
    for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
        im = cv2.imread(filename)
        name = filename.replace(src_dir, '')
        tag_name = name.replace('.jpg', '')
        tag_name = tag_name.replace('\\', '')
        rot = rotate(im, tag_name, df)
        rot_h = rotate_horizontal_and_resize(rot, (30, 75))
        tag_id_new = int(tag_name)
        selection = df[df.tag_id == tag_id_new]

        for idx, row in selection.iterrows():
            if str(row.general_class) == "small vehicle":
                add_dst = 'small vehicle/'
                if str(row.sub_class) == "sedan":
                    cv2.imwrite(dst_dir + add_dst + 'sedan/' + str(row.tag_id) + '.jpg', rot_h)
                elif str(row.sub_class) == "hatchback":
                    cv2.imwrite(dst_dir + add_dst + 'hatchback/' + str(row.tag_id) + '.jpg', rot_h)
                elif str(row.sub_class) == "minivan":
                    cv2.imwrite(dst_dir + add_dst + 'minivan/' + str(row.tag_id) + '.jpg', rot_h)
                elif str(row.sub_class) == "van":
                    cv2.imwrite(dst_dir + add_dst + 'van/' + str(row.tag_id) + '.jpg', rot_h)
                elif str(row.sub_class) == "jeep":
                    cv2.imwrite(dst_dir + add_dst + 'jeep/' + str(row.tag_id) + '.jpg', rot_h)
                elif str(row.sub_class) == "pickup":
                    cv2.imwrite(dst_dir + add_dst + 'pickup/' + str(row.tag_id) + '.jpg', rot_h)
        br += 1
        print(br)

    # preparing data for predictions
    size = (30, 75)
    X_eval = list()
    y_eval = list()
    X_tag = list()

    # hatchback
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/hatchback/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/hatchback/' + files[i], size))
        y_eval.append(0)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # jeep
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/jeep/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/jeep/' + files[i], size))
        y_eval.append(1)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # minivan
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/minivan/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/minivan/' + files[i], size))
        y_eval.append(2)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # pickup
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/pickup/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/pickup/' + files[i], size))
        y_eval.append(3)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # sedan
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/sedan/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/sedan/' + files[i], size))
        y_eval.append(4)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # van
    files = os.listdir('data2/masked_rotated_h_val/small vehicle/van/')
    files.sort()

    for i in range(0, len(files)):
        X_eval.append(transform_image('data2/masked_rotated_h_val/small vehicle/van/' + files[i], size))
        y_eval.append(5)
        tag = files[i].replace('.jpg', '')
        X_tag.append(int(tag))
    # stacking the arrays
    X_eval = np.vstack(X_eval)

    labels_index = {0: "hatchback", 1: "jeep", 2: "minivan", 3: "pickup", 4: "sedan", 5: "van"}

    # load the model
    cnn_classifier = tf.keras.models.load_model('data2/vehicle_classification_model_dropout.h5')

    cnn_pred = cnn_classifier.predict_classes(X_eval, batch_size=32)

    pretty_cm(cnn_pred, y_eval, labels_index)
    correctly_classified_indices, misclassified_indices = evaluation_indices(cnn_pred, y_eval)

    plt.figure(figsize=(36, 6))
    shuffle(correctly_classified_indices)
    plt.show()

    for plot_index, good_index in enumerate(correctly_classified_indices[0:5]):
        plt.subplot(1, 5, plot_index + 1)
        plt.imshow(X_eval[good_index])
        plt.title('Predicted: {}, Actual: {}'.format(labels_index[cnn_pred[good_index]],
                                                     labels_index[y_eval[good_index]]), fontsize=5)
    plt.show()

    print(X_tag)
    print(list(cnn_pred))
    print(y_eval)
    predicted_dict = dict(zip(X_tag, list(cnn_pred)))
    actual_dict = dict(zip(X_tag, y_eval))

    print(len(X_tag))
    print(len(list(cnn_pred)))
    print(len(y_eval))

    df = pd.read_csv('data2/val_csv.csv', sep=',')
    path = 'data2/val'
    src_dir = path
    dst_dir = 'data2/labeled_predicted_val'
    for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
        im = cv2.imread(filename)
        name = filename.replace(src_dir, '')
        img_name = name.replace('.jpg', '')
        img_name = img_name.replace('\\', '')
        labeled_img_p = label_vehicle(im.copy(), img_name, df, predicted_dict)
        cv2.imwrite(dst_dir + name, labeled_img_p)
        labeled_img_a = label_vehicle(im.copy(), img_name, df, actual_dict)
        cv2.imwrite('data2/labeled_actual_val' + name, labeled_img_a)

    report(y_eval, list(cnn_pred), labels_index)
    print("Done")


if __name__ == "__main__":
    main('val')
