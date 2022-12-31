# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM
"""
Score: 1.6368
"""

import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

proj = "/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round2"
path = f"{proj}/data/images/"
outp1 = f"{proj}/out/season2/t00001/"
outp2 = f"{proj}/out/season2/t00001/images/"

noise = cv.imread(f"{proj}/data/157823154929841251578231549202.png")
noise = cv.resize(noise, (299, 299))

data = pd.read_csv(f"{proj}/data/dev.csv")


def f_score(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))


for ipt in tqdm(os.listdir(path)):
    pin = f"{path}{ipt}"
    otp = f"{outp2}{ipt}"
    img0 = cv.imread(pin)

    d_img, k_img = 9999, ipt

    di = data[data["ImageId"] == ipt]["TargetClass"].values[0]
    dj = [i[0] for i in data[data["TrueLabel"] == di][["ImageId"]].values]

    if dj:
        pass
    else:
        di = data[data["ImageId"] == ipt]["TrueLabel"].values[0]
        dj = [i[0] for i in data[data["TrueLabel"] != di][["ImageId"]].values]

    for jpt in tqdm(dj):
        if ipt != jpt:
            img1 = cv.imread(f"{path}{jpt}")
            img1 = cv.add(noise, img1)
            s_img = f_score(img0, img1)

            if s_img < d_img:
                d_img = s_img
                k_img = jpt

    img1 = cv.imread(f"{path}{k_img}")
    img = np.clip(cv.add(noise, img1), img0-32, img0+32)
    cv.imwrite(otp, img)
    # cv.imwrite(otp.replace(".png", "_0.png"), img0)
    # cv.imwrite(otp.replace(".png", "_1.png"), img1)
