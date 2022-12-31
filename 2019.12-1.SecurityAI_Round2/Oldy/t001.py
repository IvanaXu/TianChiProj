# -*-coding:utf-8-*-
# @auth ivan 
# @time 20191204 10:43:07
# @goal test 

print(1)


# import sys
# import dlib
# import cv2 as cv
# import numpy as np
# import pandas as pd

# print("-"*100, sys.argv[1])

# path = "/Volumes/ESSD/SecurityAI_Round1/Data/images/"
# outp = f"{sys.argv[1]}"
# print("Out:", outp)

# # st, ed = 1, 712
# st, ed = 1, 100
# sall = ed-st+1
# x, y, z = 112, 112, 3
# hig, low = 1, 0

# modelv, n, nct = "_68", 15, None

# # 人脸分类器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(f"../Model/shape_predictor{modelv}_face_landmarks.dat")


# def whimax(tdata, tmin=-25, tmax=+25, tuse=10):
#     """
#         0 mean:   94.24, min:0, max:255
#         1 mean:   94.07, min:0, max:255
#         2 mean:   94.10, min:0, max:255

#         [95, -95, 0] 15 0.15
#     """
#     base = np.random.randint(0, 1, size=tdata.shape)
#     # base[:] = [-94.24*0.99999, -94.07*0.99999, -94.10*0.99999]
#     # base[:] = [-94.24, -94.07, -94.10]
#     base[:] = [-94.24, -94.07, -94.10]
#     # base[:] = [95, -95, 0]

#     rn = []
#     for n in range(tmin*tuse, tmax*tuse+1):
#         n = n/tuse
#         tdata1 = tdata.copy()
#         tdata1 = tdata1 + n
#         rn.append([n, np.array([tdata1, base]).var()])
#     rn = pd.DataFrame(rn)
#     rn.sort_values(by=1, inplace=True)
#     return int(rn.head(1).iloc[0][0])


# ndets1, ndets2 = {}, {}
# for i in range(st, ed+1):
#     pi = f"{path}"+f"00000{i}"[-5:]+".jpg"
#     po = f"{outp}/images/"+f"00000{i}"[-5:]+".jpg"
#     print(i, pi, po)

#     img = cv.imread(pi)
#     imgo = img.copy()

#     # TODO: //1//
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.imwrite(po, gray)

#     dets = detector(gray, 1)
#     ndets1[i] = dets

#     for face in dets:
#         shape = predictor(img, face)
#         for pt in shape.parts():
#             # print(pt.x, pt.y)
#             # FTODO: //2// how to eval nct
#             imgo[pt.y-n:pt.y+n, pt.x-n:pt.x+n] = img[pt.y-n:pt.y+n, pt.x-n:pt.x+n] + whimax(img[pt.y-n:pt.y+n, pt.x-n:pt.x+n])
#             # imgo[pt.y-n:pt.y+n, pt.x-n:pt.x+n] = -255
    
#     def f(n):
#         return [
#             (y//2-y//n, x//2-x//n),  
#             (y//2-y//n, x//2),
#             (y//2-y//n, x//2+x//n),  
#             (y//2, x//2-x//n),
#             (y//2, x//2+x//n),
#             (y//2+y//n, x//2-x//n),
#             (y//2+y//n, x//2),
#             (y//2+y//n, x//2+x//n),
#         ]
    
#     # dets = None
#     if not dets:
#         # FTODO: //3// IF NONE.
#         for pypx in [(y//2, x//2)]+f(4)+f(8)+f(16):
#             (py, px) = pypx
#             py, px = int(py), int(px)
#             imgo[py-n:py+n, px-n:px+n] = img[py-n:py+n, px-n:px+n] + whimax(img[pt.y-n:pt.y+n, pt.x-n:pt.x+n])
    
#     cv.imwrite(po, imgo)
#     # break

# for i in range(st, ed+1):
#     po = f"{outp}/images/"+f"00000{i}"[-5:]+".jpg"
#     pot = f"{po[:-4]}A{po[-4:]}"
#     print(i, po, pot)
#     img = cv.imread(po)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     dets = detector(gray, 1)
#     ndets2[i] = dets

#     for face in dets:
#         shape = predictor(img, face)
#         for pt in shape.parts():
#             pt_pos = (pt.x, pt.y)
#             cv.circle(img, pt_pos, 2, (0, 255, 0), 1)
#         cv.imwrite(pot, img)

# lndets1 = sum([0 if i else 1 for i in ndets1.values()])
# lndets2 = sum([0 if i else 1 for i in ndets2.values()])
# dets12 = sum([1 if i!=j else 0 for i,j in zip(ndets1.values(), ndets2.values())])

# print(
#     "Result", "\n",
#     "ndets1:%4d, %.4f\n" % (lndets1, lndets1/sall), # ndets1, "\n",
#     "ndets2:%4d, %.4f\n" % (lndets2, lndets2/sall), # ndets2, "\n",
#     "dets12:%4d, %.4f\n" % (dets12, dets12/sall)
# )



