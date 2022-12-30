#
import os
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

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

def slength(f1, f2):
    return np.sqrt(np.sum(np.square(f1 - f2)))

def cal(imgX, imgY, name):
    imgY = imgX + imgY
    imgY[np.where(imgY <= imin)] = imin
    imgY[np.where(imgY >= imax)] = imax

    cv.imwrite(f"{odir}/{i_img}", imgY)
    imgY = cv.imread(f"{odir}/{i_img}")
    
    imgsl = slength(imgX, imgY)

    img = cv.hconcat([imgX, imgY])
    cv.imwrite(f"{tdir}/bads.jpg", img)
    
    img = keras.preprocessing.image.load_img(
        f"{tdir}/bads.jpg", target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print(score, np.max(score[0]), np.max(score[1]))

    return imgsl, class_names[np.argmax(score)], 100*np.max(score[0])


for i_img in tqdm(lpng):
    img0 = cv.imread(f"{ldir}/{i_img}")

    n = 0
    while True:
        _img = np.random.randint(-Big2, Big2, size=(wh, wh, 3))
        rimg = cal(imgX=img0, imgY=_img, name=i_img)
        # print(rimg)
        
        n += 1
        if rimg[2] <= 50 or n > 20:
            break
    break



