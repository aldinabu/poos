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


def report(y_eval, cnn_pred, labels_index):
    # one vs all approach
    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity = []
    specificity = []
    accuracy = []
    precision = []

    for key in labels_index.keys():
        for label, predict in zip(y_eval, cnn_pred):
            if key == label and key == predict:
                tp += 1
            elif key != label and key != predict:
                tn += 1
            elif key != label and key == predict:
                fp += 1
            elif key == label and key != predict:
                fn += 1
        sens = tp / (tp + fn)
        sensitivity.append(sens)
        spec = tn / (tn + fp)
        specificity.append(spec)
        acc = (tp + tn) / (tp + tn + fp + fn)
        accuracy.append(acc)
        ppv = tp / (tp + fp)
        precision.append(ppv)
        tp, tn, fp, fn = 0, 0, 0, 0
    print(sensitivity)
    print(specificity)
    print(accuracy)
    print(precision)
    for key, label in labels_index.items():
        print(label + ' sensitivity: {0:.2f}%'.format(sensitivity[key]))
        print(label + ' specificity: {0:.2f}%'.format(specificity[key]))
        print(label + ' accuracy: {0:.2f}%'.format(accuracy[key]))
        print(label + ' precision: {0:.2f}%'.format(precision[key]))
