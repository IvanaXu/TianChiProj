# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

import sys
import dlib
import cv2 as cv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# print("-"*100, sys.argv[1])

path = "/data/gproj/code/SecurityAI_Round1/Data/images/"
outp = f"{sys.argv[1]}"
# print("Out:", outp)

st, ed = 1, 712
# st, ed = 1, 100
sall = ed-st+1
x, y, z = 112, 112, 3
hig, low = 1, 0
A, B, C = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

modelv, n, nct = "_68", 19, None

# 人脸分类器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../Model/shape_predictor{modelv}_face_landmarks.dat")


def whimax(tdata, y, x, tmin=-25, tmax=+25):
    """
        0 mean:   94.24, min:0, max:255
        1 mean:   94.07, min:0, max:255
        2 mean:   94.10, min:0, max:255

        [95, -95, 0] 15 0.15
    """
    base1 = np.random.randint(0, 1, size=tdata.shape)
    base2 = np.random.randint(0, 1, size=tdata.shape)
    base3 = np.random.randint(0, 1, size=tdata.shape)
    # base4 = np.random.randint(0, 1, size=tdata.shape)
    base5 = np.random.randint(0, 1, size=tdata.shape)

    base1[:] = [-94.24, -94.07, -94.10]
    base2[:] = [y, x, 255]
    base3[:] = [255, x, y]
    # base4[:] = [tdata[:,:,0].mean(), tdata[:,:,1].mean(), tdata[:,:,2].mean()]
    base5[:] = [A, B, C]
    
    rn = []
    for n in range(tmin, tmax+1):
        tdata1 = tdata.copy()
        tdata1 = tdata1 + n
        rn.append([
            n, np.array([
                tdata1, 
                # base1,
                # base2,
                # base3,
                # base4,
                base5
            ]).var()
        ])
    rn = pd.DataFrame(rn)
    rn.sort_values(by=1, inplace=True)
    return int(rn.head(1).iloc[0][0])


ndets1, ndets2 = {}, {}
for i in range(st, ed+1):
    pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
    po = f"{outp}/images/"+f"00000{i}"[-5:]+".jpg"
    print(i, pi, po)

    img = cv.imread(pi)
    imgo = img.copy()

    # TODO: //1//
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    det_s = detector(gray, 1)
    ndets1[i] = det_s

    for face in det_s:
        shape = predictor(img, face)
        ni = 0
        for pt in shape.parts():
            ni += 1
            if 0 <= pt.x < x and 0 <= pt.y < y:
                imgo[pt.y-n:pt.y+n, pt.x-n:pt.x+n] = img[pt.y-n:pt.y+n, pt.x-n:pt.x+n] + whimax(img[pt.y-n:pt.y+n, pt.x-n:pt.x+n], pt.y, pt.x)

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

    if not det_s:
        for pypx in [(y//2, x//2)]+f(4)+f(8)+f(16):
            (py, px) = pypx
            py, px = int(py), int(px)
            if 0 <= px < x and 0 <= py < y:
                imgo[py-n:py+n, px-n:px+n] = img[py-n:py+n, px-n:px+n] + whimax(img[py-n:py+n, px-n:px+n], py, px)

    cv.imwrite(po, imgo)
    # break

for i in range(st, ed+1):
    po = f"{outp}/images/"+f"00000{i}"[-5:]+".jpg"
    pot = f"{outp}/timages/A"+f"00000{i}"[-5:]+".jpg"
    # print(i, po, pot)
    img = cv.imread(po)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    det_s = detector(gray, 1)
    ndets2[i] = det_s

    for face in det_s:
        shape = predictor(img, face)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv.circle(img, pt_pos, 2, (0, 255, 0), 1)
        cv.imwrite(pot, img)


lndets1 = sum([0 if i else 1 for i in ndets1.values()])
lndets2 = sum([0 if i else 1 for i in ndets2.values()])
dets12 = sum([1 if i!=j else 0 for i,j in zip(ndets1.values(), ndets2.values())])

result = f"Result {n}\n ndets1: {lndets1}, {lndets1/sall}\n ndets2: {lndets2}, {lndets2/sall}\n dets12: {dets12}/{sall}, {dets12/sall}\n"
with open("../Out/result", "w") as f:
    f.write(result)



