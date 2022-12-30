# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM
"""
8, 112, 38.3654, no clip
"""

import cv2 as cv
import numpy as np

path = "/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round1/Data/images/"
o_path = "/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round1/Out/testRA/images/"

st, ed = 1, 712
x0, x1 = 8, 112 # 38.3654

for i in range(st, ed+1):
    pn = f"00000{i}"[-5:]
    pin = f"{path}{pn}.jpg"
    otp = f"{o_path}{pn}.jpg"
    print("-"*100, "\n", pn, pin, otp)

    img = cv.imread(pin)
    img0 = img.copy()

    img = cv.resize(img, (x0, x0), cv.INTER_LINEAR)
    img = cv.resize(img, (x1, x1), cv.INTER_LINEAR)

    cv.imwrite(otp, img)



