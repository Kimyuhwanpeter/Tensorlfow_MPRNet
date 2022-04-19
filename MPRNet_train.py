# -*- coding:utf-8 -*-
from MPRNet_model import *
from random import random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,

                           #"tr_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/low_light/",

                           #"tr_lab_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/aug_train_images/",
                           
                           #"tr_txt_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/train_fix.txt",

                           "tr_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/low_light/",

                           "tr_lab_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/",
                           
                           "tr_txt_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/train.txt",
                           
                           "batch_size": 1,
                           
                           "epochs": 50,
                           
                           "lr": 0.0001,

                           "train": True,

                           "sample_images": "C:/Users/Yuhwan/Downloads/sample_images",
                           
                           "save_checkpoint": "C:/Users/Yuhwan/Downloads/checkpoint",

                           "pre_checkpoint": False,

                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/checkpoint",
                           
                           "te_img_path": "D:/[1]DB/[5]4th_paper_DB/crop_weed/CropWeed Field Image Dataset (CWFID)/dataset-1.0/low_light/",
                           
                           "test_images": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1 = 0.5)

def CharbonnierLoss(target, predict):

    diff = target - predict
    loss = tf.reduce_mean(tf.math.sqrt((diff * diff) + (0.001 * 0.001)))
    return loss

def EdgeLoss(target, predict):

    k = tf.constant([[0.05, 0.25, 0.4, 0.25, 0.05]], tf.float32)
    k_t = tf.linalg.matrix_transpose(k)
    kernel = tf.linalg.matmul(k_t, k)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    target_img = tf.pad(target, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    target_filtered = tf.nn.depthwise_conv2d(target_img, kernel, strides=[1,1,1,1], padding="VALID")
    target_down = target_filtered[:, ::2, ::2, :]
    target_new_filter = tf.zeros_like(target_filtered).numpy()
    target_new_filter[:, ::2, ::2, :] = target_down*4
    target_new_filter = tf.pad(target_new_filter, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    target_filtered = tf.nn.depthwise_conv2d(target_new_filter, kernel, strides=[1,1,1,1], padding="VALID")
    target_diff = target - target_filtered 


    predict_img = tf.pad(predict, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    predict_filtered = tf.nn.depthwise_conv2d(predict_img, kernel, strides=[1,1,1,1], padding="VALID")
    predict_down = predict_filtered[:, ::2, ::2, :]
    predict_new_filter = tf.zeros_like(predict_filtered).numpy()
    predict_new_filter[:, ::2, ::2, :] = predict_down*4
    predict_new_filter = tf.pad(predict_new_filter, [[0,0], [2,2], [2,2], [0,0]], "REFLECT")
    predict_filtered = tf.nn.depthwise_conv2d(predict_new_filter, kernel, strides=[1,1,1,1], padding="VALID")
    predict_diff = predict - predict_filtered 


    loss = CharbonnierLoss(target_diff, predict_diff)

    # https://github.com/swz30/MPRNet/blob/main/Denoising/losses.py

    return loss

def tr_func(img_data, lab_data):

    img = tf.io.read_file(img_data)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    #img = tf.image.random_brightness(img, max_delta=50.)
    #img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    #img = tf.image.random_hue(img, max_delta=0.2)
    #img = tf.image.random_contrast(img, lower=1., upper=2.)
    #img = tf.clip_by_value(img, 0, 255)
    #img = tf.image.per_image_standardization(img)
    img = (img - tf.keras.backend.min(img)) / (tf.keras.backend.max(img) - tf.keras.backend.min(img))

    lab = tf.io.read_file(lab_data)
    lab = tf.image.decode_png(lab, 3)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size])
    #lab = tf.image.random_brightness(lab, max_delta=50.)
    #lab = tf.image.random_saturation(lab, lower=0.5, upper=1.5)
    #lab = tf.image.random_hue(lab, max_delta=0.2)
    #lab = tf.image.random_contrast(lab, lower=1., upper=2.)
    #lab = tf.clip_by_value(lab, 0, 255)
    #lab = tf.image.per_image_standardization(lab)
    lab = (lab - tf.keras.backend.min(lab)) / (tf.keras.backend.max(lab) - tf.keras.backend.min(lab))

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, lab

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:

        restored = run_model(model, images, True)
        stage_3, stage_2, stage_1 = restored
        
        stage_3 = (stage_3 - tf.keras.backend.min(stage_3)) / (tf.keras.backend.max(stage_3) - tf.keras.backend.min(stage_3))
        stage_2 = (stage_2 - tf.keras.backend.min(stage_2)) / (tf.keras.backend.max(stage_2) - tf.keras.backend.min(stage_2))
        stage_1 = (stage_1 - tf.keras.backend.min(stage_1)) / (tf.keras.backend.max(stage_1) - tf.keras.backend.min(stage_1))
        #stage_3 = tf.clip_by_value(stage_3, 0, 1)
        #stage_2 = tf.clip_by_value(stage_2, 0, 1)
        #stage_1 = tf.clip_by_value(stage_1, 0, 1)s

        stage_loss = EdgeLoss(labels, stage_3)
        stage_loss += EdgeLoss(labels, stage_2)
        stage_loss += EdgeLoss(labels, stage_1)

    grads = tape.gradient(stage_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return stage_loss

def main():
    
    model = MPRNet(inputs_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model,optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    data_list = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
    img_data = [FLAGS.tr_img_path + data for data in data_list]
    img_data = np.array(img_data)
    lab_data = [FLAGS.tr_lab_path + data for data in data_list]
    lab_data = np.array(lab_data)
    
    if FLAGS.train:
        count = 0
        for epoch in range(FLAGS.epochs):
            tr_gener = tf.data.Dataset.from_tensor_slices((img_data, lab_data))
            tr_gener = tr_gener.shuffle(len(img_data))
            tr_gener = tr_gener.map(tr_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(img_data) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels)

                if count % 10 == 0:
                    print("epoch: {} MPR loss = {} [{}/{}]".format(epoch, loss, step+1, tr_idx))


                if count % 100 == 0:
                    restored = run_model(model, batch_images, False)
                    stage_3, _, _ = restored

                    for i in range(FLAGS.batch_size):
                        original_image = batch_images[i].numpy()
                        restored_image = stage_3[i].numpy()
                        restored_image = (restored_image - np.min(restored_image)) / (np.max(restored_image) - np.min(restored_image))

                        plt.imsave(FLAGS.sample_images + "/original_image_{}step_{}.png".format(count, i), original_image * 0.5 + 0.5)
                        plt.imsave(FLAGS.sample_images + "/predict_image_{}step_{}.png".format(count, i), restored_image)


                count += 1

            ckpt = tf.train.Checkpoint(model=model,optim=optim)
            ckpt.save(FLAGS.save_checkpoint + "/" + "MPR_Net.ckpt")

if __name__ == "__main__":
    main()
