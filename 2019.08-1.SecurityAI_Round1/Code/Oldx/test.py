# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM

import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(pi):
    pi = tf.cast(pi, tf.float32)
    pi = pi/255
    pi = tf.image.resize(pi, (224, 224))
    pi = pi[None, ...]
    return pi


# Helper function to extract labels from probability vector
def get_imagenet_label(x):
    return decode_predictions(x, top=1)[0][0]


def create_adversarial_pattern(input_image, input_label):
    print(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        print(input_label, prediction)
        loss = loss_object(input_label, prediction)
        print(loss)
        
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def display_images(pi, description):
    _, i_label, i_confidence = get_imagenet_label(pretrained_model.predict(pi))
    print('{} {} : {:.2f}% Confidence'.format(description, i_label, i_confidence * 100))
    return i_confidence

path = "/data/gproj/code/SecurityAI_Round1/Data/images/"
o_path = "/data/gproj/code/SecurityAI_Round1/Out/testR/images/"
# st, ed = 1, 712
st, ed = 1, 1

for i in range(st, ed+1):
    pn = f"00000{i}"[-5:]
    pin = f"{path}{pn}.jpg"
    otp = f"{o_path}{pn}.jpg"
    print("-"*100, "\n", pn, pin, otp)
    t = 999

    img = cv.imread(pin)
    image_raw = tf.io.read_file(pin)
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)
    print(image_probs, pretrained_model(image))

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    labrador_retriever_index = 208
    label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
    # print(image_probs, image_probs.shape, label)
    perturbations = create_adversarial_pattern(image, label)
    # plt.imshow(perturbations[0])
    # plt.savefig("../Out/testR/test002.jpg")

    epsilons = [i/100 for i in range(0, 100, 20)]
    # epsilons = [0, 0.5]
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

    for ei, eps in enumerate(epsilons):
        adv_x = image + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, 0, 1)
        t_ = display_images(adv_x, descriptions[ei])
        pi = tf.image.resize(adv_x, (112, 112))
        if t_ < t:
            cv.imwrite(otp, np.clip(pi[0].numpy(), img-25, img+25))
            t = t_

