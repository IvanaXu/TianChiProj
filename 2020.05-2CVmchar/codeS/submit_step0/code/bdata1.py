# -*-coding:utf-8-*-
# @auth ivan
# @time 2020-06-13 11:20:50
# @goal v0.1


import os
import json
import cv2 as cv
from tqdm import tqdm
from pycocotools.coco import COCO


typ1, typ2 = "train", "test"
dat_path = "/tcdata"
out_path = "/myspace"

out_dict = {}
out_dict["info"] = {
    "description": "COCO CVMCHAR Dataset",
    "url": "/",
    "version": "0.1",
    "year": 2020,
    "contributor": "IVAN",
    "date_created": "2020/02/02"
}
out_dict["licenses"] = [
    {
        "url": "/",
        "id": 1,
        "name": "CVMCHAR License"
    }
]


# class
with open(f"{dat_path}/mchar_{typ1}.json", "r") as f:
    data = f.readlines()
    json_file = data[0]
j_obj = json.loads(json_file)
# print(j_obj)


classl = {}
imagel, imagen = [], 0
annotl, annotn = [], 0


#
for p0 in tqdm(os.listdir(f"{dat_path}/mchar_{typ1}")[:100], desc="bdata0"):
    p1 = j_obj[p0]
    img = cv.imread(f"{dat_path}/mchar_{typ1}/{p0}")
    # print("\n", p0, p1, img.shape)

    imagel.append({
        "license": 1,
        "file_name": f"{dat_path}/mchar_{typ1}/{p0}",
        "coco_url": p0,
        "width": img.shape[0],
        "height": img.shape[1],
        "date_captured": "2020-02-02 01:23:45",
        "flickr_url": "/",
        "id": imagen
    })
    
    for top, height, left, width, label in zip(p1["top"], p1["height"], p1["left"], p1["width"], p1["label"]):
        if label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            # print(top, height, left, width, label)
            top, height, left, width = int(top), int(height), int(left), int(width)
            top, height, left, width = max(top,0), max(height,0), max(left,0), max(width,0)
            
            if label not in classl:
                classl[label] = label

            annotl.append({
                "segmentation": [],
                "area": width * height, 
                "iscrowd": 0,
                "image_id": imagen, 
                "bbox": [left, top, width, height],
                "category_id": label, 
                "id": annotn
            })

            annotn += 1
    imagen += 1
    # break

# print(classl, classn)
# print(imagel, imagen)
# print(annotl, annotn)


print(classl)
with open(f'{out_path}/class_{typ2}.json', "w") as f:
    for classk, classv in classl.items():
        f.write(f"000{classv}"[-3:]+f",{classk}\n")

out_dict["categories"] = [
    {
        "id": classv, 
        "name": f"000{classv}"[-3:], 
        "supercategory": "clothes"
    } for classk, classv in classl.items()
]
out_dict["images"] = imagel
out_dict["annotations"] = annotl


#
out_json = f'{out_path}/annotations_{typ2}.json'
target = json.dumps(out_dict, ensure_ascii=False)
with open(out_json, 'w') as f:
    f.write(target)

# 
try:
    print(">>> TEST COCO.")
    coco = COCO(out_json)
except Exceptions as e:
    print(e)


    
