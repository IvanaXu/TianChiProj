# 
import os
import time
import oss2
import json
import pickle
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkfacebody.request.v20191230.CompareFaceRequest import CompareFaceRequest

access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', '')
access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', '')
bucket_name = os.getenv('OSS_TEST_BUCKET', 'ivan-bucket-out-002')
endpoint = os.getenv('OSS_TEST_ENDPOINT', 'oss-cn-shanghai.aliyuncs.com')

bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

client = AcsClient(access_key_id, access_key_secret, 'cn-shanghai')


#
def _f1(imgA=None, imgB=None, Rect=None):
    while True:
        try:
            time.sleep(0.68)
            if imgA:
                oss2.resumable_upload(bucket, 'imgA.jpg', imgA)
            if imgB:
                oss2.resumable_upload(bucket, 'imgB.jpg', imgB)

            request = CompareFaceRequest()
            request.set_accept_format('json')

            request.set_ImageURLA("https://"+bucket_name+"."+endpoint+"/imgA.jpg")
            request.set_ImageURLB("https://"+bucket_name+"."+endpoint+"/imgB.jpg")

            response = client.do_action_with_exception(request)
            
            if Rect:
                return json.loads(response)["Data"]["Confidence"], json.loads(response)["Data"]["RectAList"]
            else:
                return json.loads(response)["Data"]["Confidence"]
        except Exception as e:
            print(e)


#
def _f2(imgA, imgB):
    imgA, imgB = cv.imread(imgA), cv.imread(imgB)
    return np.sqrt(np.sum(np.square(imgA - imgB)))


#
r = lambda : f"{random.randint(11111, 99999)}_{random.randint(11111, 99999)}.jpg"


ldir = "../data/images/"
cdir = "../outs/comprs/"
tdir = "../temp/"
lpng = [_ for _ in os.listdir(ldir) if ".jpg" in _][:]

N = 10
wh = 112
Big1 = 125
Big2 = 25
Big3 = 5
cut = 0.50

print("Test1", _f1(imgA=f'{ldir}00001.jpg', imgB=f'{ldir}00001.jpg'))
print("Test2", _f1(imgB=f'{ldir}00002.jpg'))

os.system(f"rm  -rf {cdir}/like/* {cdir}/nolike/*")

result = {}
for i_img in tqdm(lpng):
    img0 = cv.imread(f"{ldir}/{i_img}")
    _f1(imgA=f"{ldir}/{i_img}", imgB=f"{ldir}/{i_img}")

    for j_img in tqdm(lpng):
        name = r()
        img1 = cv.imread(f"{ldir}/{j_img}")
        img = cv.hconcat([img0, img1])

        if i_img == j_img:
            cv.imwrite(f"{cdir}/like/{name}", img)
            result[name] = 1
        else:
            if random.random() > N/len(lpng):
                continue

            cv.imwrite(f"{cdir}/nolike/{name}", img)
            result[name] = 0

    
    for _ in tqdm(range(N)):
        name = r()
        img1 = img0 + np.random.randint(-Big1, Big1, size=(wh, wh, 3))
        cv.imwrite(f"{tdir}/bads.jpg", img1)
        img1 = cv.imread(f"{tdir}/bads.jpg")
        img = cv.hconcat([img0, img1])

        if _f1(imgB=f"{tdir}/bads.jpg") > cut:
            cv.imwrite(f"{cdir}/like/{name}", img)
            result[name] = 1
        else:
            cv.imwrite(f"{cdir}/nolike/{name}", img)
            result[name] = 0
    
    for _ in tqdm(range(N)):
        name = r()
        img1 = img0 + np.random.randint(-Big2, Big2, size=(wh, wh, 3))
        cv.imwrite(f"{tdir}/bads.jpg", img1)
        img1 = cv.imread(f"{tdir}/bads.jpg")
        img = cv.hconcat([img0, img1])

        if _f1(imgB=f"{tdir}/bads.jpg") > cut:
            cv.imwrite(f"{cdir}/like/{name}", img)
            result[name] = 1
        else:
            cv.imwrite(f"{cdir}/nolike/{name}", img)
            result[name] = 0

    for _ in tqdm(range(N)):
        name = r()
        img1 = img0 + np.random.randint(-Big3, Big3, size=(wh, wh, 3))
        cv.imwrite(f"{tdir}/bads.jpg", img1)
        img1 = cv.imread(f"{tdir}/bads.jpg")
        img = cv.hconcat([img0, img1])

        if _f1(imgB=f"{tdir}/bads.jpg") > cut:
            cv.imwrite(f"{cdir}/like/{name}", img)
            result[name] = 1
        else:
            cv.imwrite(f"{cdir}/nolike/{name}", img)
            result[name] = 0
    # break

print(result)
with open(f"{cdir}/result.pkl", "wb") as f:
    pickle.dump(result, f, -1)



