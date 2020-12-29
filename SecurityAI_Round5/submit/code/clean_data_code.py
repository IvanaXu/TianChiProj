#
import os
import sys
import json
import random
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

dt = "../s2_data/data/"
ot = "../user_data/tmp_data/"
"""
/outs/work, json
/outs/temp0, train mask full
/outs/temp1, train full
/outs/temp2, train mask rectangle
/outs/temp3, test full
"""

os.system(f"rm -rf {ot}/work")
os.system(f"rm -rf {ot}/temp0")
os.system(f"rm -rf {ot}/temp1")
os.system(f"rm -rf {ot}/temp2")
os.system(f"rm -rf {ot}/temp3")

os.system(f"mkdir -p {ot}/work")
os.system(f"mkdir -p {ot}/temp0")
os.system(f"mkdir -p {ot}/temp1")
os.system(f"mkdir -p {ot}/temp2")
os.system(f"mkdir -p {ot}/temp3")


N = 1550
P = 125


# 
classl = {"cheat": 0, "pmiss": 1}

rjb = {
    "info": {
        "description": "COCO SecurityAI_Round5 Dataset",
        "url": "/",
        "version": "0.1",
        "year": 2020,
        "contributor": "IVAN",
        "date_created": "2020/10/24"
    },
    "licenses": [
        {
            "url": "/",
            "id": 1,
            "name": "SecurityAI_Round5 License"
        }
    ],
    "categories": [
        {
            "id": classv, 
            "name": f"000{classv}"[-3:], 
            "supercategory": classk
        } for classk, classv in classl.items()
    ]
}
print(rjb)




nnn = 0
annotN = 0
dflip = {0:"N", 1:-1, 2:0, 3:1, 4:"M2", 5:"MF", 6:"blur", 7:"chicken", 8:"jely"}

# record: imagel, imagen, annotl, annotn
record = {}
recordp = "/user_data/tmp_data/temp1"
for _n in range(3):
    record[f"imagel{_n}"] = []
    record[f"annotl{_n}"] = []


# train
for _i, i in tqdm(enumerate(os.listdir(f"{dt}/train_mask"))):
    if ".png" in i and sys.argv[1] == "train":
        i = i.replace(".png", "")
        # TODO: 
        _n = random.choice(list("0000000012"))
        
        for _itype in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            l1, l2 = [], []
            _type = dflip[_itype]
            
            img0 = np.random.randint(0, 1, size=(N,N,3), dtype=np.uint8)
            img0[:, :] = P

            imgb = cv.imread(f"{dt}/train/{i}.jpg")
            imgm = cv.imread(f"{dt}/train_mask/{i}.png")
            i_y, i_x, _ = imgm.shape

            if _type in [-1, 0, 1]:
                imgb = cv.flip(imgb, _type)
                imgm = cv.flip(imgm, _type)
            elif _type == "chicken":
                g = 2
                
                i_yg, i_xg = i_y//g, i_x//g
                c_y = random.randint(i_yg, i_y)
                c_x = random.randint(i_xg, i_x)
                imgb = imgb[:c_y, :c_x]
                imgm = imgm[:c_y, :c_x]
                i_y, i_x, _ = imgm.shape
            elif _type == "blur":
                # imgb = cv.blur(imgb, (2, 2))
                g = 2
                
                i_yg, i_xg = i_y//g, i_x//g
                _imgb = imgb.copy()
                _imgm = imgm.copy()
                imgb[0:i_yg, 0:i_xg] = _imgb[i_yg:i_yg*g, i_xg:i_xg*g]
                imgb[i_yg:i_yg*g, i_xg:i_xg*g] = _imgb[0:i_yg, 0:i_xg]
                imgm[0:i_yg, 0:i_xg] = _imgm[i_yg:i_yg*g, i_xg:i_xg*g]
                imgm[i_yg:i_yg*g, i_xg:i_xg*g] = _imgm[0:i_yg, 0:i_xg]
            elif _type == "jely":
                g = 2
                
                i_yg, i_xg = i_y//g, i_x//g
                _imgb = imgb.copy()
                _imgm = imgm.copy()
                imgb[0:i_yg, 0:i_xg] = _imgb[i_yg:i_yg*g, i_xg:i_xg*g]
                imgb[i_yg:i_yg*g, i_xg:i_xg*g] = _imgb[0:i_yg, 0:i_xg]
                imgm[0:i_yg, 0:i_xg] = _imgm[i_yg:i_yg*g, i_xg:i_xg*g]
                imgm[i_yg:i_yg*g, i_xg:i_xg*g] = _imgm[0:i_yg, 0:i_xg]
            elif _type == "M2":
                g = 2
                
                i_yg, i_xg = i_y//g, i_x//g
                c_y = random.randint(0, i_y-i_yg)
                c_x = random.randint(0, i_x-i_xg)
                
                imgb[c_y:c_y+i_yg, c_x:c_x+i_xg] = P
                imgm[c_y:c_y+i_yg, c_x:c_x+i_xg] = 0
                l1 = [[c_x, c_y, i_xg, i_yg]]
            elif _type == "MF":
                g = 16
                
                i_yg, i_xg = i_y//g, i_x//g
                c_y = random.randint(0, i_y-i_yg)
                c_x = random.randint(0, i_x-i_xg)
                
                imgb[c_y:c_y+i_yg, c_x:c_x+i_xg] = P
                imgm[c_y:c_y+i_yg, c_x:c_x+i_xg] = 0
                l1 = [[c_x, c_y, i_xg, i_yg]]
            
            
            # save full jpg
            img0[:i_y, :i_x] = imgm
            # cv.imwrite(f"{ot}/temp0/_{i}_{_type}.jpg", img0)

            img1 = img0.copy()
            img1[:i_y, :i_x] = imgb
            cv.imwrite(f"{ot}/temp1/_{i}_{_type}.jpg", img1)


            # cut the contours
            imgray = cv.cvtColor(imgm, cv.COLOR_BGR2GRAY)  
            ret, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY) 
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            img2 = img0.copy()
            img2[::] = 255
            cv.drawContours(img2, contours, -1, (255,0,0), 10)


            # get the rectangle list
            l1 += [[0, i_y, N, N-i_y], [i_x, 0, N-i_x, N]]
            l2 += [list(cv.boundingRect(_)) for _ in contours]
            
            # draw the rectangle list
            for _x, _y, _w, _h in l1:
                cv.rectangle(img2, (_x,_y), (_x+_w,_y+_h), (0,0,255), 5) 

            for _x, _y, _w, _h in l2:
                cv.rectangle(img2, (_x,_y), (_x+_w,_y+_h), (0,255,0), 5) 

            # cv.imwrite(f"{ot}/temp2/_{i}_{_type}.jpg", img2)


            # json: imagel, imagen, annotl, annotn
            _i_ = _i*10 + _itype
            record[f"imagel{_n}"].append({
                "license": 1,
                "file_name": f"{recordp}/_{i}_{_type}.jpg",
                "coco_url": f"_{i}_{_type}.jpg",
                "width": N,
                "height": N,
                "date_captured": "2020-02-02 01:23:45",
                "flickr_url": "/",
                "id": _i_
            })


            for b_obj in l1:
                record[f"annotl{_n}"].append({
                    "segmentation": [],
                    "area": b_obj[2] * b_obj[3],
                    "iscrowd": 0,
                    "image_id": _i_,
                    "bbox": b_obj,
                    "category_id": classl["pmiss"], 
                    "id": annotN
                })
                annotN += 1

            for b_obj in l2:
                record[f"annotl{_n}"].append({
                    "segmentation": [],
                    "area": b_obj[2] * b_obj[3],
                    "iscrowd": 0,
                    "image_id": _i_,
                    "bbox": b_obj,
                    "category_id": classl["cheat"], 
                    "id": annotN
                })
                annotN += 1
        
        nnn += 1
        if nnn > 9999:
            break


# 0, 1, 2: "trai", "test", "vals"
rj0 = rjb.copy()
rj0["images"] = record[f"imagel0"]
rj0["annotations"] = record[f"annotl0"]

rj1 = rjb.copy()
rj1["images"] = record[f"imagel1"]
rj1["annotations"] = record[f"annotl1"]

rj2 = rjb.copy()
rj2["images"] = record[f"imagel2"]
rj2["annotations"] = record[f"annotl2"]


# 
_T = {"trai": rj0, "test": rj1, "vals": rj2}

for _t in _T.keys():
    with open(f"{ot}/work/class_{_t}.json", "w") as f:
        for classk, classv in classl.items():
            f.write(f"000{classv}"[-3:]+","+classk+"\n")

for _tk, _tv in _T.items():
    target = json.dumps(_tv, ensure_ascii=False)
    with open(f"{ot}/work/annotations_{_tk}.json", 'w') as f:
        f.write(target)


# test
nnn = 0
test_ot = f"{ot}/temp3"
for i in tqdm(os.listdir(f"{dt}/test")):
    if ".jpg" in i:
        i = i.replace(".jpg", "")
        img0 = np.random.randint(0, 1, size=(N,N,3), dtype=np.uint8)
        img0[:, :] = P
        
        imgb = cv.imread(f"{dt}/test/{i}.jpg")
        i_y, i_x, _ = imgb.shape
        
        img0[:i_y, :i_x] = imgb
        cv.imwrite(f"{test_ot}/_{i}.jpg", img0)
        
        nnn += 1
        if nnn > 9999:
            break



