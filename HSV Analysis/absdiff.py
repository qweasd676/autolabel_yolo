import cv2
import numpy as np
import os 

def grayto3D(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.expand_dims(img_gray,axis = 2)
    img_gray= np.concatenate((img_gray,img_gray,img_gray),axis= -1)
    return img_gray

def contour2(img,src_img,name):

    img_gray = img[:,:,0]
    draw_img = src_img.copy()
    contours, hierarchy = cv2.findContours(img_gray.copy(), cv2.RETR_TREE  ,cv2.CHAIN_APPROX_NONE)

    for j in contours:
        area = cv2.contourArea(j)
        if area > 5000 : 
            (x,y,w,h) = cv2.boundingRect(j)
            cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,0,255),2)
            # cv2.fillPoly(img, [j], (255,255,255))
            crop_img_rgb = src_img[y:y+h, x:x+w]
            crop_img_gray = img[y:y+h, x:x+w]
            segmentation =cv2.bitwise_and(crop_img_rgb ,crop_img_gray)   
            cv2.imwrite('./label/{0}.png'.format(name[:-4]),segmentation)
            # cv2.drawContours(draw_img,j, -1, (0, 0, 0),5)
            # cv2.imshow('img',cv2.resize(img,(600,600)))
            # cv2.waitKey(0)

            break

    return draw_img

def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        if(filename.endswith('.jpg') or filename.endswith('.JPG')):
            #print(filename) #just for test
            #img is used to store the image data 
            img = cv2.imread(directory_name + "/" + filename)
            img_gray = grayto3D(img)

            res = cv2.absdiff(bg_gray,img_gray)
            (T,res) = cv2.threshold(res, 50, 255, cv2.THRESH_BINARY)
            res = cv2.erode(res, None, iterations=1)
            res = cv2.dilate(res, None, iterations=36)
            res = cv2.erode(res, None, iterations=36)
            res = cv2.dilate(res, None, iterations=12)

            print(res.shape)

            segmentation =cv2.bitwise_and(img ,res)   

            # cv2.imshow('gray_res{0}'.format(str(filename)),cv2.resize(res,(600,600)))
            

        #     cv2.imshow('gray{0}'.format(str(filename)),cv2.resize(res,(600,600)))
            draw_img = contour2(res,img,filename)
            cv2.imshow('{0}'.format(str(filename)),cv2.resize(draw_img,(600,600)))
            cv2.waitKey(0)







bg = cv2.imread('./image1/bg/1.jpg')
bg_gray = grayto3D(bg.copy())

# cv2.imshow('aa',bg)
# cv2.waitKey(0)
read_directory("image1")
cv2.destroyAllWindows()