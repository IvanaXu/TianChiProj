#
import os
import cv2 as cv
from tqdm import tqdm

st, el = 1, 712

pNum = "T000001"
proj = ".."
pimg = f"{proj}/Data/images"
pout = f"{proj}/Out/{pNum}"

savl = []
for i in tqdm(range(st, el+1)):
    pi = f"{pimg}/"+f"00000{i}"[-5:]+".jpg"
    op = f"{pout}/images/"+f"00000{i}"[-5:]+".jpg"
    
    if not os.path.exists(op):
        img = cv.imread(pi)
        cv.imwrite(op, img)
        savl.append(1)
print(len(savl))


