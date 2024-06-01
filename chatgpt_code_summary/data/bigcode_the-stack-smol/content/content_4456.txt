import cv2
import os,shutil
import numpy as np
from Adb import Adb
import time

class Photo():
    '''
        提取图片信息，比较图片
    '''
    def __init__(self,img_path) -> None:
        '''
            读取图片
        '''
        self.img = cv2.imread(img_path)


    

class sourceData():
    '''
        获取测试数据
    '''
    def __init__(self) -> None:
        pass

    @staticmethod
    def getScreenPhoto():
        adb =  Adb(device='d5c42b2a')
        for x in range(100):
            adb.screenCap()
            adb.pullBackScreenCap(os.path.join('.','photo',time.strftime("%Y-%m-%d_%H-%M-%S.png", time.localtime()) ))
            print("截图",time.asctime(time.localtime()))
            time.sleep(3)

    @staticmethod
    def calcOujilide(img):
        img_new = img[938:1035,1935:2247]
        img_new_num = np.sum(img_new)/(img_new.shape[0]*img_new.shape[1]*img_new.shape[2])
        return img_new_num
    
    @staticmethod
    def calcFangcha(img):
        '''
            计算938:1035,1935:2247区域间图片的方差，用于比较图片见相似程度
            计算过程，对图像每一行像素求平均，对所有行像素平均值求方差
            return (int)
        '''
        img_new = img[938:1013,1935:2247]
        img_avg = np.mean(img_new,axis=(0,2))
        return np.var(img_avg)


if __name__ is '__main__':
    static_num = sourceData.calcFangcha(cv2.imread(os.path.join("adb","screen.png")))
    for img_name in os.listdir(os.path.join("photo")):
        img = cv2.imread(os.path.join("photo",img_name))
        img_num = sourceData.calcFangcha(img)
        chazhi = abs(static_num-img_num)
        # chazhi = (abs(static_num**2-img_num**2))**0.5
        print(img_name,"的差值为",chazhi)
        if chazhi<20:
            print("Copy this file: ",img_name)
            shutil.copyfile(os.path.join("photo",img_name),os.path.join("photo2",img_name))
            print("Write this file: ",img_name)
            cv2.imwrite(os.path.join("photo3",img_name),img[938:1013,1935:2247])


    # '''截图 400s'''
    # sourceData.getScreenPhoto()