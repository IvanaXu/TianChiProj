# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191225 22:43:07
# @goal test A base jpeg

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
print("-"*100, sys.argv[1])

path = "/Volumes/ESSD/SecurityAI_Round2/Data"
ipath = f"{path}/images"
outp = f"{sys.argv[1]}"

x, y = 299, 299
a, b = 320, 320

# ---------------------------------------
for i, _, k in os.walk(ipath):
    for pt in k:
        pi = f"{i}{os.path.sep}{pt}"
        po = f"{outp}/images/{pt}"
        print(pi, po)

        img = cv.imread("../Out/base.jpeg")
        img = cv.resize(img, (a, b), cv.INTER_LINEAR)
        img = cv.resize(img, (x, y), cv.INTER_LINEAR)
        cv.imwrite(po, img)
# ---------------------------------------



