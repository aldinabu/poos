from skimage import exposure
import cv2
import glob
import os


def clahehist(img):  # for histogram equalization
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def gamma_correction(img):  # for brigthening (gamma < 1) && contrast (gamma > 1)
    # Gamma
    gamma_corrected = exposure.adjust_gamma(img, 0.5)
    return gamma_corrected


def logarithmic_correction(img):  # for brigthening/contrast
    # Logarithmic
    logarithmic_corrected = exposure.adjust_log(img, 1)
    return logarithmic_corrected


def sigmoid_correction(img):  # for contrast
    # Sigmoid
    sigmoid = exposure.adjust_sigmoid(img, 0.5, 10, False)
    return sigmoid


print("Working on it, please have patience")
src_dir = 'dataset/training_imagery'
dst_dir = 'dataset/filtered_training/'

for filename in glob.glob(os.path.join(src_dir, '*')):
    im = cv2.imread(filename)
    name = filename.replace(src_dir, '')

    bright = gamma_correction(im)
    contrast = sigmoid_correction(bright)
    histogram = clahehist(contrast)
    cv2.imwrite(dst_dir + name, histogram)

    # cl1 = np.hstack((im, bright, contrast, histogram))  # stacking images side-by-side
    # cv2.imwrite(dst_dir + name + c, cl1)
print("Done")
