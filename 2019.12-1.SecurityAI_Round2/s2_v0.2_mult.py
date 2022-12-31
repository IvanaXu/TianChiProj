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
from multiprocessing import Pool


proj = "/Users/ivan/Desktop/ALL/Code/PyProduct/GitProj/gproj/code/SecurityAI_Round2"
path = f"{proj}/data/images/"
outp1 = f"{proj}/out/season2/t00001/"
outp2 = f"{proj}/out/season2/t00001/images/"

noise = cv.imread(f"{proj}/data/157823154929841251578231549202.png")
noise = cv.resize(noise, (299, 299))

data = pd.read_csv(f"{proj}/data/dev.csv")


def f_score(f1, f2):
    return np.sqrt(np.sum(np.square(f1-f2)))


def f(num, ipt):
    img_0 = cv.imread(f"{path}{ipt}")
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
            img_1 = cv.imread(f"{path}{jpt}")
            img_1 = 125-cv.addWeighted(noise, 1, img_1, 0.5, 0)
            s_img = f_score(img_0, img_1)

            if s_img < d_img:
                d_img = s_img
                k_img = jpt
                # break
    print(num, ipt)
    return {ipt: k_img}


if __name__ == '__main__':
    pool = Pool(processes=6)
    result = []
    for n, i in enumerate(os.listdir(path)):
        result.append(pool.apply_async(f, args=(n, i, )))

    pool.close()
    pool.join()

    r_ipt = {}
    for i in result:
        r_ipt.update(i.get())

    for ipt in tqdm(os.listdir(path)):
        pin = f"{path}{ipt}"
        otp = f"{outp2}{ipt}"

        img0 = cv.imread(pin)
        img1 = cv.imread(f"{path}{r_ipt[ipt]}")
        img = np.clip(125-cv.addWeighted(noise, 1, img1, 0.5, 0), img0-32, img0+32)
        cv.imwrite(otp, img)
