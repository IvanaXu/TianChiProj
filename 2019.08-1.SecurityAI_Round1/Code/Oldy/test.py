# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

import dlib
import pandas as pd
import numpy as np
import cv2 as cv

path = "/Volumes/ESSD/SecurityAI_Round1/Data/images/"
tp = "/Volumes/ESSD/SecurityAI_Round1/Data/images/00001.jpg"
op = "/Volumes/ESSD/SecurityAI_Round1/Data/out/demo/"
img = cv.imread(tp)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 人脸分类器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Model/shape_predictor_68_face_landmarks.dat")
dets = detector(gray, 1)


# 寻找人脸的标定点
for face in dets:
    shape = predictor(img, face)
    for pt in shape.parts():
        print(pt.x, pt.y)
        img[pt.y-5:pt.y+5, pt.x-5:pt.x+5] = img[pt.y-5:pt.y+5, pt.x-5:pt.x+5] - 25
cv.imwrite(f"{op}00001_1.jpg", img)

img = cv.imread(tp)
for face in dets:
    shape = predictor(img, face)
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv.imwrite(f"{op}00001_2.jpg", img)

iii = np.random.randint(0, 1, size=(112, 112, 3))
iii[:] = [155, 155, 155]
cv.imwrite(f"{op}00001_3.jpg", iii)


st, ed = 1, 712
crn = []
for i in range(st, ed+1):
    pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
    img = cv.imread(pi)
    print(i, pi)
    crn.append(img)
crn = np.array(crn)
print(
    crn.shape, "\n",
    f"mean:{crn[:,:,0].mean():{8}.{4}}, min:{crn[:,:,0].min()}, max:{crn[:,:,0].max()}\n",
    f"mean:{crn[:,:,1].mean():{8}.{4}}, min:{crn[:,:,1].min()}, max:{crn[:,:,1].max()}\n",
    f"mean:{crn[:,:,2].mean():{8}.{4}}, min:{crn[:,:,2].min()}, max:{crn[:,:,2].max()}\n",
)


