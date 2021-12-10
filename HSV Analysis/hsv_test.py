import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mplimg
import numpy as np
# import pywt
# from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

# from skimage.restoration import (denoise_wavelet,estimate_sigma)
# from skimage.util import random_noise
# from skimage.metrics import peak_signal_noise_ratio
# import skimage.io
# from skimage.restoration import (denoise_wavelet, estimate_sigma)
# from skimage.metrics import peak_signal_noise_ratio as PSNR
# import math

class hsv_parmeter:
    lower = np.array([90, 110 , 50])  #Red lower limit
    upper = np.array([109, 255, 255])  #Red upper limit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    def __init__(self,img):
        # pdata_plt_hsv = cv2.resize(pdata_plt_hsv,(600,600))

        #self.img = cv2.resize(img,(600,600))
        self.img1 = img
        self.img = self.img1.copy()

        self.hsv_transformation()

    def hsv_transformation(self):

        # change to hsv model
        hsv_pdata = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        cv2.imshow('image1',cv2.resize(hsv_pdata,[600,600]))

        # get mask
        hsv_mask  = ~(cv2.inRange(hsv_pdata,self.lower,self.upper))
        cv2.imshow('hsv_mask',cv2.resize(hsv_mask,[600,600]))
        cv2.imwrite('hsv_mask.jpg',hsv_mask)

        # detect blue
        hsv_res  = cv2.bitwise_and(self.img1,self.img1,mask= hsv_mask)
        # plt BGR to RGB
        # pdata_plt_hsv = cv2.cvtColor(hsv_res, cv2.COLOR_HSV2BGR)
        #pdata_plt_hsv = cv2.resize(pdata_plt_hsv,(600,600))

        #rgb_pdata = cv2.cvtColor(pdata_plt_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('image',cv2.resize(hsv_res,[600,600]))
        cv2.imwrite('image.png',hsv_res)

img1 = cv2.imread("./image/34.jpg")
test = hsv_parmeter(img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
