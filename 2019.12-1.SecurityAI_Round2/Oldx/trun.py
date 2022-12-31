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
outp = f"{sys.argv[1]}"

num = 0
N = 9999

dt = pd.read_csv(f"{path}/dev.csv")
dt.index = dt["ImageId"]
ldt = dt[["TrueLabel", "TargetClass"]].to_dict()
dtrue = ldt["TrueLabel"]
dtarg = ldt["TargetClass"]
# print(dtrue, dtarg)

rlist = {}
for i, _, k in os.walk(f"{path}/images"):
    for pt in k:
        pi = f"{i}{os.path.sep}{pt}"
        pg = dtrue[pt]
        num += 1
        print(num, pi, pt, pg)

        img = cv.imread(pi)

        if pg not in rlist:
            rlist[pg] = [img]
        else:
            rlist[pg].append(img)
        
        if num > N:
            break

f = open(f"{path}/dev_T.csv", "w")
f.write("TrueLabel,Mean0,Mean1,Mean2\n")

for i, j in rlist.items():
    k = "%d,%.4f,%.4f,%.4f\n" % (i, np.array(j)[:,:,0].mean()-np.array(j)[:,:,0].std(), np.array(j)[:,:,1].mean()+np.array(j)[:,:,1].std(), np.array(j)[:,:,2].mean())
    f.write(k)

f.close()

dtT = pd.read_csv(f"{path}/dev_T.csv")
print(dtT.head())

