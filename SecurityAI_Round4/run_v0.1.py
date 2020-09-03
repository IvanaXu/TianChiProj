#
import os
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm

dt = "/Users/ivan/Desktop/ALL/Data/SecurityAI_Round4"
ot = f"{dt}/outs/T000002/images"

N = 130
for iTN in range(N, N+10):
    os.system(f"rm -rf {dt}/outs/T000002/images/*")

    def fpoinnt(l, deep):
        deep = 2 ** deep
        _ = [i for i in range(0, l, l//deep) if i > 0]
        _ = [i for n, i in enumerate(_) if n%2 == 0]
        _ = [(i,j) for i in _ for j in _]
        return _
    TN = iTN
    TR = random.randint(111111,999999)
    (x, y) = fpoinnt(500,5)[TN]


    for i in tqdm(os.listdir(f"{dt}/data/images")):
        img = cv.imread(f"{dt}/data/images/{i}")

        img = cv.circle(img, (x,y),  38, (255,0,0), 1)
        img = cv.circle(img, (x,y),  44, (0,0,243), 1)
        img = cv.circle(img, (x,y),  72, (255,0,0), 1)
        img = cv.circle(img, (x,y), 108, (255,0,0), 1)
        img = cv.circle(img, (x,y), 145, (255,0,0), 1)
        
        cv.imwrite(f"{ot}/{i}", img)
        # break

    os.system(f"sh {dt}/outs/T000002/run.sh")
    os.system(f"cp {dt}/outs/T000002/images.zip {dt}/outs/T000002/images_{TN}_{TR}.zip")
