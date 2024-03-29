# -*-coding:utf-8-*-
# @auth ivan
# @time 20191118 08:55:32 
# @goal test read jpg

import sys
import os
import pathlib
import random
import time
import tensorflow as tf
import matplotlib.pyplot as plt
print("-"*100, tf.__version__, sys.argv[1])

N, M = 32, 200
Epochs = 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT, IMG_WIDTH = 600, 600

para = str(sys.argv[1])

with open("Cervical_Cancer/list/neg.list", "r") as rf:
    r = rf.readlines()
neg = [p.strip("\n") for p in r]
with open("Cervical_Cancer/list/pos_t.list", "r") as rf:
    r = rf.readlines()
pos_t = [p.strip("\n") for p in r]
with open("Cervical_Cancer/list/pos.list", "r") as rf:
    r = rf.readlines()
pos = [p.strip("\n") for p in r]
with open("Cervical_Cancer/list/black.list", "r") as rf:
    r = rf.readlines()
black = [p.strip("\n") for p in r]

if not os.path.exists("Cervical_Cancer/model/"+para):
    os.system("mkdir Cervical_Cancer/model/"+para)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

acc, loss = [], []

for mm in range(M):
    print("."*100, mm)
    time.sleep(1)
    
    random.shuffle(neg)
    random.shuffle(pos_t)
    random.shuffle(pos)
    random.shuffle(black)

    # all_image_paths = neg[:N] + pos_t[:N] + black[:N]
    all_image_paths = neg[:N] + pos_t[:N] + pos[:N] + black[:N]

    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    steps_per_epoch = tf.math.ceil(image_count/BATCH_SIZE).numpy()

    label_to_index = {"neg":0, "black":0, "pos_t":1, "pos":1}
    all_image_labels = [label_to_index[pathlib.Path(path).parent.parent.name] for path in all_image_paths]
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    default_timeit_steps = 2 * steps_per_epoch + 1

    def testit(ds, steps=default_timeit_steps):
        print(ds)

        overall_start = time.time()
        # 在开始计时之前
        # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
        it = iter(ds.take(steps+1))
        next(it)
        
        start = time.time()
        for i, (images,labels) in enumerate(it):
            if i%10 == 0:
                print('.',end='')
                # print(images, labels)
        print()
        end = time.time()

        duration = end - start
        print("{} batches: {} s".format(steps, duration))
        print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))
        print("Total time: {}s".format(end-overall_start))

    # # preprocess_image  = lambda image: tf.image.decode_jpeg(image)
    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        return image
    load_and_preprocess_image = lambda path: preprocess_image(tf.io.read_file(path))


    # 1
    paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = paths_ds.map(load_and_preprocess_image)
    ds = image_ds.map(tf.io.serialize_tensor)

    def parse(x):
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        result = tf.reshape(result, [IMG_HEIGHT, IMG_WIDTH, 3])
        return result
    ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

    # 3
    ds = tf.data.Dataset.zip((ds, label_ds))
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # history = model.fit(ds, epochs=Epochs, steps_per_epoch=3)
    history = model.fit(ds, epochs=Epochs, steps_per_epoch=default_timeit_steps)
    model.save("Cervical_Cancer/model/"+para) 

    acc += history.history['accuracy']
    loss += history.history['loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.ylim([0.5, 1])
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.subplot(1, 2, 2)
plt.ylim([0, 1])
plt.plot(epochs_range, loss, label='Training Loss')
plt.savefig("Cervical_Cancer/model/"+para+"/model.png")
plt.show()



