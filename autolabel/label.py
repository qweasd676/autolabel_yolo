#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import random
from tqdm import tqdm  #跑時間條的library


# In[2]:


def rotate_bound(image, angle):
    """

    :param image: 原圖
    :param angle: 旋轉角度
    :return: 旋轉後的圖
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img


# In[11]:


class com:
    def __init__(self, background_list, cut_list):  
        self.background_list = background_list
        self.cut_list = cut_list
        

        
    def back_init(self, file_name):    #重新讀取新的背景圖，一些參數更新
        self.background = cv2.imread("./pocky_background/"+background_list[random.randint(0,2)])
        self.Xrange = []
        self.Yrange = []
        self.file_name = file_name
        
    def cut_init(self):   #重新讀取新的裁切圖
        self.classid = random.randint(0,4)
        self.cut = cv2.imread("./cut/"+cut_list[self.classid])
        
        
    def check_cover(self, Xmin, Xmax, Ymin, Ymax):    #確認有沒有重疊覆蓋，生成一堆的可以忽略
        
        for i in range(len(self.Xrange)):
            
            Xmin_check = 0
            Xmax_check = 0
            Ymin_check = 0
            Ymax_check = 0
            
            if (Xmin >= self.Xrange[i][0]) and (Xmin <= self.Xrange[i][1]):
                Xmin_check = 1
            if (Xmax >= self.Xrange[i][0]) and (Xmax <= self.Xrange[i][1]):
                Xmax_check = 1
            if (Ymin >= self.Yrange[i][0]) and (Ymin <= self.Yrange[i][1]):
                Ymin_check = 1
            if (Ymax >= self.Yrange[i][0]) and (Ymax <= self.Yrange[i][1]):
                Ymax_check = 1
                
            if (self.Xrange[i][0] >= Xmin) and (self.Xrange[i][0] <= Xmax):
                Xmin_check = 1
            if (self.Xrange[i][1] >= Xmin) and (self.Xrange[i][1] <= Xmax):
                Xmax_check = 1
            if (self.Yrange[i][0] >= Ymin) and (self.Yrange[i][0] <= Ymax):
                Ymin_check = 1
            if (self.Yrange[i][1] >= Ymin) and (self.Yrange[i][1] <= Ymax):
                Ymax_check = 1
            
            if (Xmin_check or Xmax_check) and (Ymin_check or Ymax_check):
               
                return False
        return True
        
    def combine(self):     #合成
        angle = random.randrange(0,360)
        shift_x = random.randint(0,150)
        shift_y = random.randint(0,400)
        rimg = rotate_bound(self.cut, angle)
        
        Xmin = shift_x
        Xmax = shift_x+rimg.shape[0]
        Ymin = shift_y
        Ymax = shift_y+rimg.shape[1]
        
        #if check_over: (確認重疊)
        if True:   
            X = (Xmin + (Xmax - Xmin)/2) / self.background.shape[0]
            Y = (Ymin + (Ymax - Ymin)/2) / self.background.shape[1]
            W = (Xmax - Xmin)/self.background.shape[0]
            H = (Ymax - Ymin)/self.background.shape[1]
            label =str(self.classid) + " {:.6f} {:.6f} {:.6f} {:.6f}\n".format(Y,X,H,W)    #Yolo的格式

            for i in range(rimg.shape[0]):
                for j in range(rimg.shape[1]):
                    if rimg[i,j].all() == 0:
                        pass
                    else:
                        self.background[Xmin+i, Ymin+j] = rimg[i, j]
                        
            self.Xrange.append((Xmin, Xmax))  #確認重疊用的
            self.Yrange.append((Ymin, Ymax))  #確認重疊用的
            with open("./label/"+self.file_name+".txt", "a") as f:  #寫label檔
                f.write(label)
    def img_write(self):    #存合成圖
        cv2.imwrite("./label_img/"+self.file_name+".jpg",self.background)


# In[15]:


background_list = ["background1.jpg", "background2.jpg", "background3.jpg"]
cut_list = ["1.png","2.png","3.jpg","4.jpg","5.jpg"]

COM = com(background_list, cut_list)
for i in tqdm(range(10)):    #生成幾張
    COM.back_init(str(i))
    for j in range(1):       #一張圖裡有合成幾個裁切圖
        COM.cut_init()
        COM.combine()
    COM.img_write()

