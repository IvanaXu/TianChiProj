# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
print("-"*100, sys.argv[1])

path = "/Volumes/ESSD/SecurityAI_Round2/Data"
ipath = f"{path}/images"
outp = f"{sys.argv[1]}"

n = 25
x, y, z = 299, 299, 3

dt = pd.read_csv(f"{path}/dev.csv")
print(dt.head())
dt.index = dt["ImageId"]
ldt = dt[["TrueLabel", "TargetClass"]].to_dict()
dtrue, dtarg = ldt["TrueLabel"], ldt["TargetClass"]

dtT = pd.read_csv(f"{path}/dev_T.csv")
print(dtT.head())
dtT.index = dtT["TrueLabel"]
ldtT = dtT[["Mean0", "Mean1", "Mean2"]].to_dict()
dmean0, dmean1, dmean2 = ldtT["Mean0"], ldtT["Mean1"], ldtT["Mean2"]
# print(dmean0, dmean1, dmean2)


def whimax(tdata, b1, b2, tmin=-32, tmax=+32, tuse=1):
    """
        mean:   112.2, min:0, max:255
        mean:   112.5, min:0, max:255
        mean:   112.7, min:0, max:255
    """
    base1 = np.random.randint(0, 1, size=tdata.shape)
    base1[:] = b1

    base2 = np.random.randint(0, 1, size=tdata.shape)
    base2[:] = b2

    base12 = np.random.randint(0, 1, size=tdata.shape)
    base12[:] = [b1[0]+b2[0], b1[1]+b2[1], b1[2]+b2[2]]

    base3 = np.random.randint(0, 1, size=tdata.shape)
    base3[:] = [-200, -200, -200]

    nmin, vmin = 0, 99999999
    for n in range(tmin, tmax+1):
        tdata1 = tdata + n
        v = np.array([tdata1,base1,base2,base12,base3]).var()
        if v < vmin:
            nmin, vmin = n, v
    return nmin


def f(n):
    return [
        (y//2-y//n, x//2-x//n),  
        (y//2-y//n, x//2),
        (y//2-y//n, x//2+x//n),  
        (y//2, x//2-x//n),
        (y//2, x//2+x//n),
        (y//2+y//n, x//2-x//n),
        (y//2+y//n, x//2),
        (y//2+y//n, x//2+x//n),
    ]

num = 0
N = 9999
# N = 10
# other = 255
other = -255


for i, _, k in os.walk(ipath):
    for pt in k:
        pi = f"{i}{os.path.sep}{pt}"
        po = f"{outp}/images/{pt}"
        pg1, pg2 = dtrue[pt], dtarg[pt]
        bm1, bm2 = [
            dmean0[pg1] if pg1 in dmean0 else other, 
            dmean1[pg1] if pg1 in dmean1 else other,
            dmean2[pg1] if pg1 in dmean2 else other
        ],[
            dmean0[pg2] if pg2 in dmean0 else other, 
            dmean1[pg2] if pg2 in dmean1 else other,
            dmean2[pg2] if pg2 in dmean2 else other
        ]
        num += 1
        print(num, pi, po, pg1, pg2, bm1, bm2)

        img = cv.imread(pi)
        imgo = img.copy()

        for pypx in [(y//2, x//2)]+f(4)+f(8)+f(16)+f(32)+f(64)+f(128)+f(0.5)+f(1.5)+f(2.5):
            (py, px) = pypx
            py, px = int(py), int(px)
            # for nn in range(-n, n+1):
            #     if 0 <= py+nn <= y and 0 <= px+nn <= x:
            #         imgo[py+nn, px+nn] = img[py+nn, px+nn] + whimax(img[py+nn, px+nn], bm1, bm2)
            imgo[py-n:py+n, px-n:px+n] = img[py-n:py+n, px-n:px+n] + whimax(img[py-n:py+n, px-n:px+n], bm1, bm2)
        cv.imwrite(po, imgo)

        if num > N:
            break



