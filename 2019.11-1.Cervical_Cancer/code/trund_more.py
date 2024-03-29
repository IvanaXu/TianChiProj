# -*-coding:utf-8-*-
# @auth ivan
# @time 20191118 08:55:32 
# @goal test read jpg

import sys
import pandas as pd
import tensorflow as tf
print("-"*100, tf.__version__, sys.argv[1])

N = 1000
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT, IMG_WIDTH = 600, 600

para = str(sys.argv[1])
df = pd.read_csv("Cervical_Cancer/list/test.list", chunksize=N, usecols=["path"])
rf = open("Cervical_Cancer/result/"+para+".log", "w")
rf.write("path,predict\n")
restored_keras_model = tf.keras.models.load_model("Cervical_Cancer/model/"+para)
n = 0


for dfi in df:
    n += 1
    print(".", n)

    all_image_paths = dfi["path"].values

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        return image
    load_and_preprocess_image = lambda path: preprocess_image(tf.io.read_file(path))

    paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = paths_ds.map(load_and_preprocess_image)
    ds = image_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    result = restored_keras_model.predict(ds, steps=1)

    for ri, rj in zip(all_image_paths, result):
        if rj[0] > 0:
            # print(ri,rj[0])
            rf.write("%s,%.8f\n" % (ri,rj[0]))
    
    # if n > 2:
    #     break
rf.close()

