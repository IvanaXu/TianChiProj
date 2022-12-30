import os
import sys
import dlib
import time
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
import face_recognition as facer
from tqdm import tqdm


pNum = "T000001"

proj = ".."
pimg = f"{proj}/Data/images"
pmdl = f"{proj}/Model"
pout = f"{proj}/Out/{pNum}"


modelv = "_81"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"{pmdl}/shape_predictor{modelv}_face_landmarks.dat")
dmtcnn = MTCNN()

st, ed = int(sys.argv[1]), int(sys.argv[2])
chose = int(sys.argv[3])
igroup, group = int(sys.argv[4]), int(sys.argv[5])


def f_lf1(img1, img2):
    img1, img2 = cv.imread(img1), cv.imread(img2)
    return np.sqrt(np.sum(np.square(img1 - img2)))


def _score(fimg):
    _img = cv.imread(fimg)
    result1 = facer.face_landmarks(_img)
    result2 = detector(cv.cvtColor(_img, cv.COLOR_BGR2GRAY), 1)
    result3 = dmtcnn.detect_faces(_img)
    
    if result1 or result2 or result3:
        return False
    else:
        return True



_N = 128
lrun = [2, 4, 8, 16, 32, 64, 128][:chose]
pl = lambda x: sorted(
    [
        (i - _N // x, i, j - _N // x, j)
        for j in range(0, _N + 1, _N // x) if j > 0
        for i in range(0, _N + 1, _N // x) if i > 0
     ]
)


badl = []
for i in tqdm(range(st, ed+1), desc=f"R{group}_All"):
    mscore = 999999
    pi = f"{pimg}/"+f"00000{i}"[-5:]+".jpg"
    op = f"{pout}/images/"+f"00000{i}"[-5:]+".jpg"
    
    if _score(pi):
        continue
        
    if i%igroup != group:
        continue
    
    for dN in tqdm([_i_ for _i_ in range(-25, 25+1) if _i_ != 0], desc=f"R{group}_Dev"):
        for _p_, M in enumerate(lrun):
            dpl = pl(M)

            if _p_ == 0:
                ldata = np.array([1 for _ in range(M ** 2)])
            else:
                t_ldata = [__i for __i, __j in zip(pl(lrun[_p_-1]), ldata) if __j == 1]
                ldata = []
                for _i in dpl:
                    _t = 0
                    for __i in t_ldata:
                        if _i[0] < __i[1] and __i[0] < _i[1] and _i[2] < __i[3] and __i[2] < _i[3]:
                            _t = 1
                            break
                    ldata.append(_t)
                ldata = np.array(ldata)

            #
            while True:
                t_ldata = ldata.copy()
                for _i in tqdm(range(M**2)):
                    if ldata[_i] == 1:
                        ldata[_i] = 0

                        img = cv.imread(pi)
                        for nldata, _ldata in enumerate(ldata):
                            if _ldata == 1:
                                i_group = dpl[nldata]
                                img[i_group[0]:i_group[1], i_group[2]:i_group[3]] = \
                                img[i_group[0]:i_group[1], i_group[2]:i_group[3]] - dN
                        cv.imwrite(f"{pout}/temp{group}.jpg", img)

                        escore = _score(f"{pout}/temp{group}.jpg")
                        if escore:
                            pass
                        else:
                            ldata[_i] = 1
                if (ldata == t_ldata).all():
                    break
                break

        #
        img = cv.imread(pi)
        for nldata, _ldata in enumerate(ldata):
            if _ldata == 1:
                i_group = dpl[nldata]
                img[i_group[0]:i_group[1], i_group[2]:i_group[3]] = \
                img[i_group[0]:i_group[1], i_group[2]:i_group[3]] - dN
        cv.imwrite(f"{pout}/temp{group}.jpg", img)

        escore = _score(f"{pout}/temp{group}.jpg")
        fscore = f_lf1(pi, f"{pout}/temp{group}.jpg")

        if escore:
            if fscore < mscore:
                print("\n", i, dN, escore, "%9.4f"%fscore)
                mscore = fscore
                os.system(f"cp {pout}/temp{group}.jpg {op}")



