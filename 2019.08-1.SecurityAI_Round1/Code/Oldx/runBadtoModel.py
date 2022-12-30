# -*-coding:utf-8-*-
# @auth ivan
# @time 20191204 10:43:07
# @goal test

import os
import sys
import dlib
import random
import cv2 as cv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

path = "/data/gproj/code/SecurityAI_Round1/Data/images/"
o_path = "/data/gproj/code/SecurityAI_Round1/Out/runBadtoModel/images/"

g0, g1 = sys.argv[1], int(sys.argv[1])
base = int(sys.argv[2])
st, ed = (g1-1)*base+1, min(g1*base,712)
high = 40
bad0 = 10
times = 300
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../Model/shape_predictor_68_face_landmarks.dat")
f = open(f"/data/gproj/code/SecurityAI_Round1/Out/runBadtoModel/temp/r{g0}", "w")

def f_xy(images):
    cv_face = detector(cv.cvtColor(images, cv.COLOR_BGR2GRAY), 1)
    if cv_face:
        result = []
        for face in cv_face:
            shape = predictor(images, face)
            for pt in shape.parts():
                y, x = pt.y, pt.x
                if 0 < x < 112 and 0 < y < 112:
                    result.append((y, x))
    else:
        result = None
    return result


for i in range(st, ed+1):
    pn = f"00000{i}"[-5:]
    pi = f"{path}{pn}.jpg"
    op = f"{o_path}{pn}.jpg"
    img = cv.imread(pi)
    fxy_img = f_xy(img)

    if fxy_img:
        rk = False
        for j in range(times):
            random.shuffle(fxy_img)

            for i_bad0 in range(1, bad0+1):
                for kz in range(-25, 25+1):
                    print(f"ID:{pn}, T:{j}, Bad0:{i_bad0}, Use:{kz}")

                    img_o = img.copy()
                    for k in fxy_img[0:high]:
                        (ky, kx) = k
                        img_o[ky-i_bad0:ky+i_bad0, kx-i_bad0:kx+i_bad0] = \
                            img_o[ky-i_bad0:ky+i_bad0, kx-i_bad0:kx+i_bad0] + kz
                    img_o = np.clip(img_o, img-25, img+25)
                    if not f_xy(img_o):
                        cv.imwrite(op, img_o)
                        f.write(f"{pn}\n")
                        rk = True

                    if rk:
                        break
                if rk:
                    break
            if rk:
                break
    if not os.path.exists(op):
        os.system(f"cp {pi} {op}")
f.close()



