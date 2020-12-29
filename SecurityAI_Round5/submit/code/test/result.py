# 
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

config_file = "code/train/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "user_data/model_data/S499.pth"
model1 = init_detector(config_file, checkpoint_file, device="cuda:0")
print("M1", model1.CLASSES)

checkpoint_file = "user_data/model_data/S488.pth"
model2 = init_detector(config_file, checkpoint_file, device="cuda:0")
print("M2", model2.CLASSES)

checkpoint_file = "user_data/model_data/S477.pth"
model3 = init_detector(config_file, checkpoint_file, device="cuda:0")
print("M3", model3.CLASSES)

good = 0
bads = 255


# 
N = 9999
p5 = "user_data/tmp_data/temp3/"
p6 = "s2_data/data/test/"
p7 = "prediction_result/images/"

dresult1, dresult2, dresult3 = {}, {}, {}

nnn = 0
for _img in tqdm(os.listdir(f"{p5}")):
    # print(_img)
    if ".jpg" in _img:
        _img = _img.replace(".jpg", "").replace("_", "")
        
        result = inference_detector(model1, f"{p5}_{_img}.jpg")
        _result = [ij for i, j in enumerate(result) if i == 0 and j.shape != (0,5) for ij in j]
        dresult1[_img] = _result
        
        result = inference_detector(model2, f"{p5}_{_img}.jpg")
        _result = [ij for i, j in enumerate(result) if i == 0 and j.shape != (0,5) for ij in j]
        dresult2[_img] = _result
        
        result = inference_detector(model3, f"{p5}_{_img}.jpg")
        _result = [ij for i, j in enumerate(result) if i == 0 and j.shape != (0,5) for ij in j]
        dresult3[_img] = _result
        
        nnn += 1
        if nnn > N:
            break


#
icut = 0.00
high1, high2 = 90000, 300 # 90000, 300
logs = "and"
r1, r2, r3 = [], [], []

nnn = 0
for _img in tqdm(os.listdir(f"{p5}")):
    if ".jpg" in _img:
        _img = _img.replace(".jpg", "").replace("_", "")
        
        
        # 
        imgb1 = cv.imread(f"{p6}{_img}.jpg")
        imgb1[:] = 0
        for _iresult in dresult1[_img]:
            ymin, ymax, xmin, xmax = int(_iresult[1]), int(_iresult[3]), int(_iresult[0]), int(_iresult[2])
            if _iresult[4] >= icut and (high1 >= (ymax-ymin)*(xmax-xmin) >= high2):
                r1.append((ymax-ymin)*(xmax-xmin))
                imgb1[ymin:ymax, xmin:xmax] = bads
        
        # 
        imgb2 = cv.imread(f"{p6}{_img}.jpg")
        imgb2[:] = 0
        for _iresult in dresult2[_img]:
            ymin, ymax, xmin, xmax = int(_iresult[1]), int(_iresult[3]), int(_iresult[0]), int(_iresult[2])
            if _iresult[4] >= icut and (high1 >= (ymax-ymin)*(xmax-xmin) >= high2):
                r2.append((ymax-ymin)*(xmax-xmin))
                imgb2[ymin:ymax, xmin:xmax] = bads
        
        # 
        imgb3 = cv.imread(f"{p6}{_img}.jpg")
        imgb3[:] = 0
        for _iresult in dresult3[_img]:
            iymin, ymax, xmin, xmax = int(_iresult[1]), int(_iresult[3]), int(_iresult[0]), int(_iresult[2])
            if _iresult[4] >= icut and (high1 >= (ymax-ymin)*(xmax-xmin) >= high2):
                r3.append((ymax-ymin)*(xmax-xmin))
                imgb3[ymin:ymax, xmin:xmax] = bads
        
        
        if logs == "and":
            imgb = np.where(np.logical_and(imgb1>0, imgb2>0, imgb3>0), imgb1, 0)
        if logs == "or":
            imgb = np.where(np.logical_or(imgb1>0, imgb2>0, imgb3>0), imgb1+imgb2+imgb3, 0)
        
        cv.imwrite(f"{p7}{_img}.png", imgb)
        
        nnn += 1
        if nnn > N:
            break



