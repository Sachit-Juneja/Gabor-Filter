import cv2
import os
import glob
import numpy as np


img_dir = "/Users/sachit/Downloads/TCB\ Dataset/traindata "
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1,0)
    data.append(img)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    ret, labels = cv2.connectedComponents(img)
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    cv2.imshow('Image_0_01.tif', labeled_img)
    cv2.waitKey()

    def build_filters():
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
            return filters

    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
            return accum

    filters=build_filters()
    res1=process(img,filters)
    cv2.imshow('result',res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("checking.tif",res1)