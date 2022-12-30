#
import os
import sys
import dlib
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

p1, p2, p3, p4 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

img_height = 112
img_width = 112*2
class_names = ['like', 'nolike']

ldir = "../data/images/"
lpng = [_ for _ in os.listdir(ldir) if ".jpg" in _][:]
data_dir = "../outs/comprs/"
keras_model_path = "../outs/model"
tdir = "../temp/"
odir = "../outs/images/"

model = tf.keras.models.load_model(keras_model_path)

modelv = "_81"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../temp/shape_predictor{modelv}_face_landmarks.dat")

#
img = keras.preprocessing.image.load_img(
    f"{data_dir}/like/"+random.choice(os.listdir(f"{data_dir}/like")), target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Test1, This image most likely belongs to >{}< with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#
img = keras.preprocessing.image.load_img(
    f"{data_dir}/nolike/"+random.choice(os.listdir(f"{data_dir}/nolike")), target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Test2, This image most likely belongs to >{}< with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


imin, imax = 0, 255
Big2 = 25
N = 100
wh = 112


def f_xy(images):
    min_y, max_y, min_x, max_x = wh, 0, wh, 0
    cv_face = detector(cv.cvtColor(images, cv.COLOR_BGR2GRAY), 1)
    if cv_face:
        for face in cv_face:
            shape = predictor(images, face)
            for pt in shape.parts():
                min_y, max_y = min(pt.y, min_y), max(pt.y, max_y)
                min_x, max_x = min(pt.x, min_x), max(pt.x, max_x)
    else:
        min_y, max_y, min_x, max_x = 0, wh, 0, wh
    
    min_y, max_y, min_x, max_x = max(min_y,0), min(max_y,wh), max(min_x,0), min(max_x,wh)
    return min_y, max_y, min_x, max_x

print(
    f_xy(cv.imread(f"{ldir}/00001.jpg")), 
    f_xy(cv.imread(f"{ldir}/00003.jpg"))
)


def slength(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))


def cal(imgX, imgY, name):
    imgY = imgX + imgY
    imgY[np.where(imgY <= imin)] = imin
    imgY[np.where(imgY >= imax)] = imax

    cv.imwrite(f"{odir}/{name}", imgY)
    imgY = cv.imread(f"{odir}/{name}")
    
    imgsl = slength(imgX, imgY)

    img = cv.hconcat([imgX, imgY])
    cv.imwrite(f"{tdir}/bads-{p1}.jpg", img)
    
    img = keras.preprocessing.image.load_img(
        f"{tdir}/bads-{p1}.jpg", target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print(score, np.max(score[0]), np.max(score[1]))
    
    return imgsl, class_names[np.argmax(score)], 100*np.max(score[1])


for i_img in tqdm(lpng[p2:p3+1]):
    if os.path.exists(f"{odir}/{i_img}"):
        continue
    img0 = cv.imread(f"{ldir}/{i_img}")
    min_yl, max_yl, min_xl, max_xl = f_xy(img0)
    _img = np.random.randint(-Big2, Big2, size=(wh, wh, 3))
    _img[::] = 0
    
    # base 
    iscore, i_n_ = 0, -Big2
    for _n_ in tqdm(range(-Big2, Big2+1)):
        _img[min_yl:max_yl, min_xl:max_xl] = _n_
        rimg = cal(imgX=img0, imgY=_img, name=i_img)

        if iscore < rimg[2]:
                iscore = rimg[2]
                i_n_ = _n_
                print(_n_, rimg)
    _img[min_yl:max_yl, min_xl:max_xl] = i_n_
    rimg = cal(imgX=img0, imgY=_img, name=i_img)
    print(">1", rimg)

    # good
    iscore = 0
    for n in tqdm(range(p4)):
        ix = random.randint(min_yl, max_yl-1)
        iy = random.randint(min_xl, max_xl-1)
        it = random.randint(0, 2)
        ic = img0[ix, iy, it]

        i_n_ = ic
        for _n_ in range(max(ic-Big2, imin), min(ic+Big2, imax)+1):
            _n_ = _n_ - ic
            _img[ix, iy, it] = _n_
            rimg = cal(imgX=img0, imgY=_img, name=i_img)

            if iscore < rimg[2]/rimg[0]:
                iscore = rimg[2]/rimg[0]
                i_n_ = _n_
                print(ix, iy, it, rimg)

        if i_n_ != Big2:
            _img[ix, iy, it] = i_n_
    
    rimg = cal(imgX=img0, imgY=_img, name=i_img)
    print(">2", rimg)
    # break



