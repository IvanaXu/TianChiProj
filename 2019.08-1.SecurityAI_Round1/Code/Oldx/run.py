# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

import sys
import cv2 as cv
import numpy as np
print("-"*100, sys.argv[1])

path = "/Volumes/ESSD/SecurityAI_Round1/Data/images/"
outp = f"{sys.argv[1]}"
print(outp)

st, ed = 1, 712
x, y, z = 112, 112, 3
hig, low = 1, 0
n = 50

for i in range(st, ed+1):
    pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
    po = f"{outp}/images/"+f"00000{i}"[-5:]+".jpg"
    img = cv.imread(pi)
    # timg = np.random.randint(low, hig, size=(x, y, z))
    # timg[x//2-n:x//2+n] = low
    timg = low+1
    print(pi, po)
    cv.imwrite(po, img+timg)
    # break



