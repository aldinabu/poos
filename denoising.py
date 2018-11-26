import cv2
import glob
import os


def denoising(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return denoised_img


def unsharp_masking(src):
    gaussian = cv2.GaussianBlur(src, (9, 9), 10.0)
    unsharp_img = cv2.addWeighted(src, 1.5, gaussian, -0.5, 0, src)
    return unsharp_img


print("Working on it, please have patience")
src_dir = 'dataset/training_imagery'
dst_dir = 'dataset/denoised_training'
dst_dir2 = 'dataset/unsharped_training'
unsharp_original_flag = False
for filename in glob.glob(os.path.join(src_dir, '*')):
    im = cv2.imread(filename)
    name = filename.replace(src_dir, '')
    denoised = denoising(im)
    if unsharp_original_flag:
        unsharped = unsharp_masking(im)
    else:
        unsharped = unsharp_masking(denoised)
    cv2.imwrite(dst_dir + name, denoised)
    cv2.imwrite(dst_dir2 + name, unsharped)
print("Done")
