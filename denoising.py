import cv2


def denoise(img):
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return denoised_img


def unsharp_masking(src):
    gaussian = cv2.GaussianBlur(src, (9, 9), 10.0)
    unsharp_img = cv2.addWeighted(src, 1.5, gaussian, -0.5, 0, src)
    return unsharp_img
