# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM
"""
Score: 0.5800
"""

import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

proj = "/root/gproj/code/SecurityAI_Round2"
path = f"{proj}/data/images/"
outp = f"{proj}/out/season2/t00002/images/"

noise = cv.imread(f"{proj}/data/157823154929841251578231549202.png")
noise = cv.resize(noise, (299, 299))

for ipt in tqdm(os.listdir(path)):
    pin = f"{path}{ipt}"
    otp = f"{outp}{ipt}"
    # print(pin, otp)

    img0 = cv.imread(pin)
    img = np.clip(noise, img0-32, img0+32)
    cv.imwrite(otp, img)



