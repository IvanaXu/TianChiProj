#
import os
import sys
import random
from tqdm import tqdm
import torch
import torchvision
import cv2 as cv
import numpy as np

print("CUDA:", torch.cuda.is_available())
ldir = "../data/images/"
odir = "../outs/images/"
cdir = "../outs/comprs/"
lpng = [_ for _ in os.listdir(ldir) if ".jpg" in _][:200]
# print(lpng)

def slength(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))

npng = {_png: i for i, _png in enumerate(lpng)}
# rpng = {_png: _png for i, _png in enumerate(lpng[::-1])}
rpng = {}
for ipng in tqdm(lpng):
    img = cv.imread(f"{ldir}/{ipng}")
    smg_key, smg_min = "", 999999

    for jpng in lpng:
        if ipng != jpng:
            jmg = cv.imread(f"{ldir}/{jpng}")
            smg = slength(img, jmg)
            if smg < smg_min:
                smg_key = jpng
                smg_min = smg
    print(ipng, smg_key, smg_min)
    rpng[ipng] = smg_key
print(npng, rpng)

import foolbox as fb

model = torchvision.models.resnet152(pretrained=True).eval()
bounds = (0, 1)
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
fmodel = fmodel.transform_bounds((0, 1))

images = np.array(
    [
        np.transpose(cv.imread(f"{ldir}/{_png}"), (2,0,1)) 
        for _png in lpng
    ]
)
images = images/255
images = torch.from_numpy(images).type(torch.FloatTensor).cuda()

labels = torch.from_numpy(
    np.array(
        [npng[i] for i in lpng]
    )
).cuda()
target_classes = torch.from_numpy(
    np.array(
        [npng[rpng[i]] for i in lpng]
    )
).cuda()
print(images.shape, labels.shape)

criterion = fb.criteria.TargetedMisclassification(target_classes)

attack = fb.attacks.L2CarliniWagnerAttack(steps=7500)

advs, _, is_adv = attack(fmodel, images, criterion, epsilons=None)

print(
    fb.utils.accuracy(fmodel, images, labels),
    fb.utils.accuracy(fmodel, advs, labels),
    fb.utils.accuracy(fmodel, advs, target_classes)
)

for n, ipng in enumerate(tqdm(lpng)):
    img = np.transpose(advs[n].cpu().numpy(), (1,2,0)) * 255
    cv.imwrite(f"{odir}/{ipng}", img)

    img0 = cv.imread(f"{ldir}/{ipng}")
    img1 = cv.imread(f"{odir}/{ipng}")
    img2 = cv.hconcat([img0, img0-img1, img1-img0, img1])
    cv.imwrite(f"{cdir}/{ipng}", img2)



