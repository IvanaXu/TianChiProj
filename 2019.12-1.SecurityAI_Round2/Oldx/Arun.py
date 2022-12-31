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

x, y, z = 299, 299, 3
g0, g1 = sys.argv[2], int(sys.argv[2])
base = int(sys.argv[3])
st, ed = (g1-1)*base+1, min(g1*base,1216)
print(st, ed)


# ---------------------------------------
# for i, _, k in os.walk(ipath):
#     for pt in k:
#         pi = f"{i}{os.path.sep}{pt}"
#         po = f"{outp}/images/{pt}"
#         print(pi, po)

#         img = cv.imread(base)
#         # img = cv.resize(img, (x, y))
#         img = cv.resize(img, (x, y), cv.INTER_LINEAR)
#         cv.imwrite(po, img)
# ---------------------------------------


dt = pd.read_csv(f"{path}/dev.csv")
dt.index = dt["ImageId"]
ldt = dt[["TrueLabel", "TargetClass"]].to_dict()
dtrue, dtarg = ldt["TrueLabel"], ldt["TargetClass"]

dtT = pd.read_csv(f"{path}/dev_T.csv")
dtT.index = dtT["TrueLabel"]
ldtT = dtT[["Mean0", "Mean1", "Mean2"]].to_dict()
dmean0, dmean1, dmean2 = ldtT["Mean0"], ldtT["Mean1"], ldtT["Mean2"]


def whimax(tdata, b1, b2, b3, tmin=-32, tmax=+32):
    """
        mean:   112.2, min:0, max:255
        mean:   112.5, min:0, max:255
        mean:   112.7, min:0, max:255
    """
    base1 = np.random.randint(0, 1, size=tdata.shape)
    base1[:] = b1

    base2 = np.random.randint(0, 1, size=tdata.shape)
    base2[:] = b2

    base3 = np.random.randint(0, 1, size=tdata.shape)
    base3[:] = b3

    nmin, vmin = 0, 99999999
    for n in range(tmin, tmax+1):
        tdata_ = tdata + n
        v = np.array([tdata_, base1, base2, base3]).var()
        if v < vmin:
            nmin, vmin = n, v
    return nmin


num = 0
N = 9999
other = -255

ibase = cv.imread("../Out/base.jpeg")
ibase = cv.resize(ibase, (x, y))


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
        
        if num >= st and num <= ed:
            print(num, pi, po, pg1, pg2, bm1, bm2)

            img = cv.imread(pi)
            imgo = img.copy()

            for py in range(0, y):
                print(num, py)
                for px in range(0, x):
                    imgo[py, px] = img[py, px] + whimax(img[py, px], bm1, bm2, ibase[py, px])

            cv.imwrite(po, imgo)
        else:
            pass
        
        if num > N:
            break



