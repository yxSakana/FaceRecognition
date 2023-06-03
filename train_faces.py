#!/usr/local/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-
# @Project name: FaceRecognition
# @Filename: train_faces.py
# @Author: sxy
# @Date: 2023-06-03 20:56

import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

import sys
import logging.config
sys.path.append("/home/sxy/.conda/envs/pytorch/lib/python3.9/site-packages/")
sys.path.append("/home/sxy/.conda/envs/dlib/lib/python3.9/site-packages/")
import tensorflow as tf

from config import CONFIG
from logConfig.logConfig import *


logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)


def getPaddingSize(img: cv2.Mat):
    """

    :param img:
    :return:
    """
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


def readData(path, h, w, img_list, lab_list):
    """

    :param path:
    :param h:
    :param w:
    :param img_list:
    :param lab_list:
    :return:
    """
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            img_list.append(img)
            lab_list.append(path)


def weightVariable(shape):
    """
    创建权重变量(使用正态分布的随机值作为初始值)
    :param shape: 指定形状
    :return:
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def biasVariable(shape):
    """
    创建偏置变量(使用正态分布的随机值作为初始值)
    :param shape: 指定形状
    :return:
    """
    return tf.Variable(tf.random_normal(shape))


def conv2d(x, W):
    """
    卷积操作
    :param x: 输入特征图
    :param W: 卷积核
    :return: 卷积结果
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def maxPool(x):
    """
    执行最大池化操作
    :param x: 输入特征图
    :return: 池化结果
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    """
    执行丢弃操作(将输入中的一些元素随机置为零，以减少过拟合)
    :param x: 丢弃的张量
    :param keep: 丢弃率(保留的元素比例)
    :return:
    """
    return tf.nn.dropout(x, keep)


def cnnLayer(l=None):
    # 第一层
    W1 = weightVariable([3, 3, 3, 32])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out


def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(10):
            # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                            feed_dict={
                                               x: batch_x,
                                               y_: batch_y,
                                               keep_prob_5: 0.5,
                                               keep_prob_75: 0.75}
                                            )
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                logger.info(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({
                        x: test_x,
                        y_: test_y,
                        keep_prob_5: 1.0,
                        keep_prob_75: 1.0
                    })
                    logger.info(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        logger.info('accuracy less 0.98, exited!')


if __name__ == '__main__':
    my_faces_path = CONFIG.get("my_faces_path")
    other_faces_path = CONFIG.get("other_faces_path")
    size = CONFIG.get("size")
    # check Dir
    os.makedirs(my_faces_path) if not os.path.exists(my_faces_path) else None
    os.makedirs(other_faces_path) if not os.path.exists(other_faces_path) else None

    imgs = []
    labs = []
    # init image data
    readData(my_faces_path, size, size, imgs, labs)
    readData(other_faces_path, size, size, imgs, labs)
    # 将图片数据与标签转换成数组
    imgs = np.array(imgs)
    labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])
    # 随机划分测试集与训练集
    train_x, test_x, train_y, test_y = train_test_split(imgs,
                                                        labs,
                                                        test_size=0.05,
                                                        random_state=random.randint(0, 100))
    # 参数：图片数据的总数，图片的高、宽、通道
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # 将数据转换成小于1的数
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    logger.info('train size:%s, test size:%s' % (len(train_x), len(test_x)))
    # 图片块，每次取100张图片
    batch_size = 100
    num_batch = len(train_x) // batch_size

    x = tf.placeholder(tf.float32, [None, size, size, 3])
    y_ = tf.placeholder(tf.float32, [None, 2])

    keep_prob_5 = tf.placeholder(tf.float32)
    keep_prob_75 = tf.placeholder(tf.float32)

    cnnTrain()
