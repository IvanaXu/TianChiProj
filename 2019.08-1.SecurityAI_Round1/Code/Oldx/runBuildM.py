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

path = "/data/gproj/code/SecurityAI_Round1/Data/images/"
o_path = "/data/gproj/code/SecurityAI_Round1/Out/test/"

# st, ed = 1, 712
st, ed = 1, 100
modelv, n = "_68", 10

# 人脸分类器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../Model/shape_predictor{modelv}_face_landmarks.dat")

for i in range(st, ed+1):
    pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
    op = f"{o_path}A"+f"00000{i}"[-5:]+".jpg"
    print(pi, op)
    img = cv.imread(pi)
    img_o = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for face in detector(gray, 1):
        shape = predictor(img, face)
        for pt in shape.parts():
            x, y = pt.x, pt.y
            if 0 <= x < 112 and 0 <= y < 112:
                img_o[y, x] = img_o[y, x] - 100
        cv.imwrite(op, img_o)



