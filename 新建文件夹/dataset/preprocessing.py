import os
import cv2
import math
import time
import torch
import numpy as np
import scipy.misc as misc
import PIL.Image as Image

from PIL import ImageEnhance, ImageOps, ImageFilter


def main():
    images_path = '/data/public/scratch/raw/ng'
    dst_path = '/data/public/scratch/raw/ng_temp'

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for parent, dirs, files in os.walk(images_path):
        for file in files:
            if file.endswith('.bmp'):
                print("> Processing file: {}...".format(file))
                kernel_size = (5, 5)
                sigma = 1.5
                img = cv2.imread(os.path.join(parent, file))
                img = cv2.GaussianBlur(img, kernel_size, sigma)

                canny = cv2.Canny(img, 30, 150)
                kernel = np.ones((5, 5), np.uint8)
                erosion = cv2.dilate(canny, kernel, iterations=1)

                cloneImage, contours, heriachy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for i, contour in enumerate(contours):
                    # print(cv2.arcLength(contour, True))
                    if cv2.arcLength(contour, True) > 2000:
                        x, y, w, h = cv2.boundingRect(contour)

                        sub_img = img[y:y + h, x:x + w, :]

                        sub_img = np.array(sub_img[:, :, ::-1], dtype=np.uint8)
                        sub_img = Image.fromarray(sub_img, mode='RGB')

                        # sub_img = ImageOps.autocontrast(sub_img)
                        # sub_img = ImageOps.equalize(sub_img)
                        # sub_img = ImageOps.autocontrast(sub_img)
                        sub_img = ImageEnhance.Brightness(sub_img).enhance(0.725)
                        # sub_img = ImageOps.autocontrast(sub_img)
                        sub_img = ImageEnhance.Contrast(sub_img).enhance(2.875)
                        sub_img = ImageEnhance.Sharpness(sub_img).enhance(1.75)

                        # sub_img = ImageOps.autocontrast(sub_img)

                        sub_img = np.array(sub_img, dtype=np.uint8).copy()
                        sub_img = np.array(sub_img[:, :, ::-1], dtype=np.uint8)

                        is_show = False
                        if is_show:
                            cv2.namedWindow("ImageOrg", cv2.WINDOW_NORMAL)
                            cv2.imshow("ImageOrg", sub_img)
                            cv2.waitKey()

                        cv2.imwrite(os.path.join(dst_path, file), sub_img)

                        ref_img = cv2.imread(os.path.join(parent, file.replace('.bmp', '.jpg')))

                        if ref_img.shape == img.shape:
                            sub_ref_img = ref_img[y:y + h, x:x + w, :]

                            cv2.imwrite(os.path.join(dst_path, file.replace('.bmp', '.jpg')), sub_ref_img)

                        mask_img = cv2.imread(os.path.join(parent, file.replace('.bmp', '.png')))

                        sub_mask_img = mask_img[y:y + h, x:x + w, :]
                        cv2.imwrite(os.path.join(dst_path, file.replace('.bmp', '.png')), sub_mask_img)

                    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

            # cv2.imwrite(os.path.join('D:\\test',file), img)
            # crop_img = img[400:800,400:800,:]
            # cv2.imshow('11',crop_img)
            # cv2.waitKey(0)


'''
size = 512,overlap=128,step = size-overlap=384
1130x685 => (step*2+size,step*1+size)=(1280,896)
'''


def crop():
    src_path = '/data/public/scratch/raw/ng_temp'
    dst_dath = '/data/public/scratch/crop/ng_temp'

    if not os.path.exists(dst_dath):
        os.makedirs(dst_dath)

    for parent, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.bmp'):
                print("> Processing file: {}...".format(file))

                img = misc.imread(os.path.join(parent, file))
                img = np.array(img, dtype=np.uint8)

                mask_img = misc.imread(os.path.join(parent, file.replace('.bmp', '.png')), mode="L")
                mask_img = np.array(mask_img, dtype=np.uint8)

                size = 512
                overlap = 128
                step = size - overlap

                h_steps = (img.shape[0] - size) / step
                h_steps = int(math.ceil(h_steps))
                w_steps = (img.shape[1] - size) / step
                w_steps = int(math.ceil(w_steps))
                print(w_steps, h_steps)

                new_h = step * h_steps + size
                new_w = step * w_steps + size
                print(new_w, new_h)

                resize_img = misc.imresize(img, (new_h, new_w), interp="bilinear")

                mask_img = Image.fromarray(mask_img, mode="L")
                mask_img = mask_img.resize((new_w, new_h), resample=Image.NEAREST)
                resize_mask_img = np.array(mask_img, dtype=np.uint8).copy()

                k = 1

                for i in range(h_steps + 1):
                    y = step * i
                    for j in range(w_steps + 1):
                        x = step * j
                        crop_img = resize_img[y:y + size, x:x + size, :]
                        new_file_name = file.replace('.bmp', '_' + str(k).zfill(2) + '.bmp')
                        cv2.imwrite(os.path.join(dst_dath, new_file_name), crop_img)

                        crop_mask_img = resize_mask_img[y:y + size, x:x + size]
                        new_mask_file_name = file.replace('.bmp', '_' + str(k).zfill(2) + '.png')
                        cv2.imwrite(os.path.join(dst_dath, new_mask_file_name), crop_mask_img)

                        k += 1


if __name__ == "__main__":
    crop()