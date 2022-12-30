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
lpng = [_ for _ in os.listdir(ldir) if ".jpg" in _]
# print(lpng)


for ipng in tqdm(lpng[int(sys.argv[1]):int(sys.argv[2])]):
    import foolbox as fb

    model = torchvision.models.resnet152(pretrained=True).eval()
    bounds = (0, 1)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))

    images = np.array(
        [np.transpose(cv.imread(f"{ldir}/{ipng}"), (2,0,1)) ] + random.choices(
            [
                np.transpose(cv.imread(f"{ldir}/{_png}"), (2,0,1)) 
                for _png in lpng if _png != ipng
            ]
        )
    )
    images = images/255
    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()

    labels = torch.from_numpy(
        np.array([1, 0])
    ).cuda()
    target_classes = torch.from_numpy(
        np.array([0, 1])
    ).cuda()
    print(images.shape, labels.shape)

    criterion = fb.criteria.TargetedMisclassification(target_classes)

    attack = fb.attacks.L2CarliniWagnerAttack(steps=100)

    advs, _, is_adv = attack(fmodel, images, criterion, epsilons=None)

    print(
        fb.utils.accuracy(fmodel, images, labels),
        fb.utils.accuracy(fmodel, advs, labels),
        fb.utils.accuracy(fmodel, advs, target_classes)
    )

    img = np.transpose(advs[0].cpu().numpy(), (1,2,0)) * 255
    cv.imwrite(f"{odir}/{ipng}", img)

    img0 = cv.imread(f"{ldir}/{ipng}")
    img1 = cv.imread(f"{odir}/{ipng}")
    cv.imwrite(f"{cdir}/{ipng}", img0*0.50+(img0-img1)*0.50)

    del fb, model, fmodel, criterion, attack



