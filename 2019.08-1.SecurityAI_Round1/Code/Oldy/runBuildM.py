# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

import sys
import dlib
import cv2 as cv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

path = "/Volumes/ESSD/SecurityAI_Round1/Data/images/"

# st, ed = 1, 712
st, ed = 1, 100
modelv, n = "_68", 10

# 人脸分类器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../Model/shape_predictor{modelv}_face_landmarks.dat")

for i in range(st, ed+1):
    pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
    img = cv.imread(pi)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dets = detector(gray, 1)   
    for face in dets:
        shape = predictor(img, face)
        print(i, len(shape.parts()))
        # for pt in shape.parts():
        #     print(i, pt.x, pt.y)

    if not dets:
        print(i, "///")



