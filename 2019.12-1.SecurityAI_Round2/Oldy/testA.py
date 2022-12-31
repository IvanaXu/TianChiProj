# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM

import os
import cv2 as cv
import numpy as np

path = "/data/gproj/code/SecurityAI_Round2/data/images/"
outp = "/data/gproj/code/SecurityAI_Round2/out/testRA/images/"


for i, _, k in os.walk(path):
    for pt in k:
        pin = f"{i}{pt}"
        otp = f"{outp}{pt}"
        print(pin, otp)
        # img = cv.resize(cv.resize(cv.imread(pin), (16, 16), cv.INTER_LINEAR), (299, 299), cv.INTER_LINEAR)

        img = cv.imread(pin)
        img = cv.resize(img, (16, 16), cv.INTER_LINEAR)
        img = cv.resize(img, (64, 64), cv.INTER_LINEAR)
        img = cv.resize(img, (128, 128), cv.INTER_LINEAR)
        img = cv.resize(img, (299, 299), cv.INTER_LINEAR)
        cv.imwrite(otp, img)



