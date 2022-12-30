# -*-coding:utf-8-*-
# @auth ivan
# @time 20191224 22:37:50
# @goal test the FGSM

import dlib
import cv2 as cv
import numpy as np
import tensorflow as tf

path = "/data/gproj/code/SecurityAI_Round1/Data/images/"
o_path = "/data/gproj/code/SecurityAI_Round1/Out/testRB/images/"
st, ed, edm = 1, 1, 10
M = 68
B = -100
name = "00AAA"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f"../Model/shape_predictor_68_face_landmarks.dat")

"""
for i in range(st, ed+1):
    pn = f"00000{i}"[-5:]
    pin = f"{path}{pn}.jpg"
    otp = f"{o_path}{pn}.jpg"
    print("-"*100, "\n", pn, pin, otp)
    # t = 999

    img = cv.imread(pin)
    image_raw = tf.io.read_file(pin)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)[None, ...]
    image_prob = i_predict(image, ty="tf")

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    label = tf.constant(np.array([1 if it == i else 0 for it in range(1, 712+1)]))
    print(label)

    perturbations = create_adversarial_pattern(image, label)

    print(perturbations)

    # epsilons = [i / 100 for i in range(0, 100, 10)]
    # # epsilons = [0, 0.5]
    # descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]
    #
    # for ei, eps in enumerate(epsilons):
    #     adv_x = image + eps * perturbations
    #     adv_x = tf.clip_by_value(adv_x, 0, 1)
    #     t_ = display_images(adv_x, descriptions[ei])
    #     # pi = tf.image.resize(adv_x, (112, 112))
    #     # if t_ < t:
    #     #     cv.imwrite(otp, np.clip(pi[0].numpy(), img-25, img+25))
    #     #     t = t_
"""


def val_face68(images, ty="np"):
    if ty == "tf":
        images = tf.cast(image, tf.uint8)
        images = images.numpy()[0]

    result = []
    cv_face = detector(cv.cvtColor(images, cv.COLOR_BGR2GRAY), 1)
    if cv_face:
        for face in cv_face:
            shape = predictor(images, face)
            for pt in shape.parts():
                result.append([pt.y, pt.x])

    for _ in range(0, M-len(result)):
        result.append([B, B])
    result.sort()

    return np.array(result, dtype=float)


r_img = {name: np.array([[B, B] for _ in range(0, M)])}
for i in range(st, edm+1):
    pn = f"00000{i}"[-5:]
    pin = f"{path}{pn}.jpg"
    print("-" * 20, pn, pin)
    r_img[pn] = val_face68(cv.imread(pin))
# print(r_img)


def i_predict(images, ty="np"):
    # [0.1, 0.2, 0.1, 0.3]
    if ty == "tf":
        images = tf.cast(image, tf.uint8)
        images = images.numpy()[0]

    lxm = []
    for ix, jx in r_img.items():
        nxm = np.mean(
            [
                np.dot(kx0, kx1)/(np.linalg.norm(kx0)*np.linalg.norm(kx1))
                for kx0, kx1 in zip(jx, images)
            ]
        )
        lxm.append(nxm)
    return np.array(lxm, dtype=float)


def get_image_net_label(x):
    # ('', 'prayer_rug', 0.14394665)
    n = 0
    nxm, ixm = name, -1
    for ix in x:
        nx = f"00000{n}"[-5:].replace("00000", name)

        if ix >= ixm:
            ixm = ix
            nxm = nx
        n += 1
    return "", nxm, ixm


def create_adversarial_pattern(input_image, input_label):
    print(f"{'-' * 10}1{'-' * 10}")
    print(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = tf.constant([i_predict(val_face68(image, ty="tf"))], dtype=tf.float32)
        print(input_label, prediction)
        loss = loss_object(input_label, prediction)
        print(loss)
    print(f"{'-' * 10}2{'-' * 10}")
    gradient = tape.gradient(loss, input_image)
    print(f"{'-' * 10}3{'-' * 10}")
    print(gradient)
    signed_grad = tf.sign(gradient)
    print(f"{'-' * 10}4{'-' * 10}")
    return signed_grad


# def display_images(pi, description):
#     _, i_label, i_confidence = get_image_net_label(i_predict(pi, ty="tf"))
#     print('{} {} : {:.2f}% Confidence'.format(description, i_label, i_confidence * 100))
#     return i_confidence


for i in range(st, ed+1):
    pn = f"00000{i}"[-5:]
    pin = f"{path}{pn}.jpg"
    otp = f"{o_path}{pn}.jpg"
    print("-" * 20, pn, pin, otp)

    img = cv.imread(pin)
    print(i_predict(val_face68(img)))
    print(get_image_net_label(i_predict(val_face68(img))))

    image_raw = tf.io.read_file(pin)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)[None, ...]

    image_prob = i_predict(val_face68(image, ty="tf"))
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    # label = image_prob
    label = tf.one_hot(0, image_prob.shape[-1])

    perturbations = create_adversarial_pattern(image, label)
    print(perturbations)



