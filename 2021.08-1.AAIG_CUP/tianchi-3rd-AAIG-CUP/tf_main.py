import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa


def train_sample(train_path):
    data = pd.read_csv(train_path, header=None)
    
    _1 = pd.value_counts(data[2])
    _2 = pd.value_counts(data[3])
    print(data.shape)
    data = data[data[1] > 0]
    print(data.shape)
    data = data[data[2].isin(_1[_1 < 150].index)]
    print(data.shape)
    data = data[data[3].isin(_2[_2 < 150].index)]
    print(data.shape)

    x = data[4].str.split(" ", expand=True).values.astype(np.float32)
    y = data[5].values
    return x, y


def build_model():
    LOSS = [
        tf.keras.losses.BinaryCrossentropy(name="Loss")
    ]

    METRICS = [
        tfa.metrics.F1Score(name="F1", num_classes=1, threshold=0.50)
    ]

    _1 = tf.keras.Input((75,))
    X = tf.keras.layers.BatchNormalization()(_1)
    for u in [150, 300, 600, 600, 300, 150, 75]:
        X = tf.keras.layers.Dropout(0.5)(
            tf.keras.layers.Dense(u, "relu")(X)
        )
    _2 = tf.keras.layers.Dense(1, "sigmoid")(X)
    model = tf.keras.Model(_1, _2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=LOSS, metrics=METRICS
    )
    model.summary()
    return model


def train(train_path, model_dir, save_name):
    trainx, trainy = train_sample(train_path)
    model = build_model()
    # model_dir
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_dir + "/frozen_model", "val_F1", save_best_only=True, mode="max"
    )
    model.fit(
        trainx,
        trainy,
        batch_size=1024,
        epochs=600,
        verbose=1,
        validation_split=0.20,
        callbacks=[checkpoint],
    )
    os.remove(model_dir + "/frozen_model/keras_metadata.pb")

