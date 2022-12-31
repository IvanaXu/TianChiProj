# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM

import sys
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g0, g1 = sys.argv[1], int(sys.argv[1])
base = int(sys.argv[2])
st, ed = (g1-1)*base+1, min(g1*base,1216)
print(st, ed)

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
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def display_images(pi, description):
    _, i_label, i_confidence = get_imagenet_label(pretrained_model.predict(pi))
    print('{} {} : {:.2f}% Confidence'.format(description, i_label, i_confidence * 100))
    return i_confidence

path = "/data/gproj/code/SecurityAI_Round2/data/images/"
outp = "/data/gproj/code/SecurityAI_Round2/out/testR/images/"


num = 0
for phi, _, k in os.walk(path):
    for pt in k:
        pin = f"{phi}{pt}"
        otp = f"{outp}{pt}"
        num += 1

        if st <= num <= ed:
            print("-"*100, "\n", num, pin, otp)
            t = 0

            img = cv.imread(pin)
            image_raw = tf.io.read_file(pin)
            image = tf.image.decode_image(image_raw)
            image = preprocess(image)
            image_probs = pretrained_model.predict(image)

            loss_object = tf.keras.losses.CategoricalCrossentropy()
            labrador_retriever_index = 208
            label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
            perturbations = create_adversarial_pattern(image, label)
            # plt.imshow(perturbations[0])
            # plt.savefig("../Out/testR/test002.jpg")

            epsilons = [i/100 for i in range(0, 100, 5)]
            epsilons.remove(0)
            # epsilons = [0, 0.5]
            descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

            for ei, eps in enumerate(epsilons):
                adv_x = image + eps * perturbations
                adv_x = tf.clip_by_value(adv_x, -32, 32)
                t_ = display_images(adv_x, descriptions[ei])
                pi = tf.image.resize(adv_x, (299, 299))
                if t_ > t:
                    cv.imwrite(otp, np.clip(pi[0].numpy(), img-32, img+32))
                    t = t_

