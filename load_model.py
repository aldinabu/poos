from tensorflow import python as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from random import shuffle
from label_vehicles import label_vehicle
import pandas as pd
import glob
import cv2
import copy


def pretty_cm(y_pred, y_truth, labels):
    # pretty implementation of a confusion matrix
    cm = metrics.confusion_matrix(y_truth, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='BuGn_r')
    # labels, title and ticks
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.title('Accuracy: {0:.2f}%'.format(metrics.accuracy_score(y_truth, y_pred)*100), size=15)
    plt.xticks(list(labels.keys()), list(labels.values()), rotation='horizontal', fontsize=10)
    plt.yticks(list(labels.keys()), list(labels.values()), rotation='horizontal', fontsize=10)


def transform_image(path, size):
    # function for transforming images into a format supported by CNN
    x = tf.keras.preprocessing.image.load_img(path, target_size=(size[0], size[1]))
    x = tf.keras.preprocessing.image.img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return x


def evaluation_indices(y_pred, y_test):
    # function for getting correctly and incorrectly classified indices
    index = 0
    correctly_classified_indices = []
    misclassified_indices = []
    for label, predict in zip(y_test, y_pred):
        if label != predict:
            misclassified_indices.append(index)
        else:
            correctly_classified_indices.append(index)
        index += 1
    return correctly_classified_indices, misclassified_indices


# preparing data for predictions
size = (30, 75)
X_eval = list()
y_eval = list()
X_tag = list()

# hatchback
files = os.listdir('data/masked_rotated_h_val/small vehicle/hatchback/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/hatchback/' + files[i], size))
    y_eval.append(0)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# jeep
files = os.listdir('data/masked_rotated_h_val/small vehicle/jeep/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/jeep/' + files[i], size))
    y_eval.append(1)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# minivan
files = os.listdir('data/masked_rotated_h_val/small vehicle/minivan/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/minivan/' + files[i], size))
    y_eval.append(2)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# pickup
files = os.listdir('data/masked_rotated_h_val/small vehicle/pickup/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/pickup/' + files[i], size))
    y_eval.append(3)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# sedan
files = os.listdir('data/masked_rotated_h_val/small vehicle/sedan/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/sedan/' + files[i], size))
    y_eval.append(4)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# van
files = os.listdir('data/masked_rotated_h_val/small vehicle/van/')
files.sort()

for i in range(0, len(files)):
    X_eval.append(transform_image('data/masked_rotated_h_val/small vehicle/van/' + files[i], size))
    y_eval.append(5)
    tag = files[i].replace('.jpg', '')
    X_tag.append(int(tag))
# stacking the arrays
X_eval = np.vstack(X_eval)

labels_index = {0: "hatchback", 1: "jeep", 2: "minivan", 3: "pickup", 4: "sedan", 5: "van"}


# load the model
cnn_classifier = tf.keras.models.load_model('vehicle_classification_model_dropout.h5')

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

df = pd.read_csv('data/val_csv.csv', sep=',')
path = 'data/val'
src_dir = path
dst_dir = 'data/labeled_predicted_val'
for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
    im = cv2.imread(filename)
    name = filename.replace(src_dir, '')
    img_name = name.replace('.jpg', '')
    img_name = img_name.replace('\\', '')
    labeled_img_p = label_vehicle(im.copy(), img_name, df, predicted_dict)
    cv2.imwrite(dst_dir + name, labeled_img_p)
    labeled_img_a = label_vehicle(im.copy(), img_name, df, actual_dict)
    cv2.imwrite('data/labeled_actual_val' + name, labeled_img_a)
print("Done")

