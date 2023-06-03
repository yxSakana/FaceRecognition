#!/usr/local/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-
# @Project name: FaceRecognition
# @Filename: GetFaceData.py
# @Author: sxy
# @Date: 2023-06-03 20:14


import cv2
import os
import random
import logging.config
import sys
sys.path.append("/home/sxy/.conda/envs/pytorch/lib/python3.9/site-packages/")
sys.path.append("/home/sxy/.conda/envs/dlib/lib/python3.9/site-packages/")

import dlib

from config import CONFIG
from logConfig.logConfig import *


logging.config.dictConfig(log_config)
logger = logging.getLogger("FaceRecognition")


# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    # image = []
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


def img_handle(img: cv2.Mat, index: int, is_relight: bool, output_dir: str):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_img, 1)  # 用detector进行人脸检测

    for i, d in enumerate(dets):
        logger.info('Being processed picture %s' % index)

        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        face = img[x1:y1, x2:y2]
        # 调整图片的对比度与亮度(都取随机数)来增加样本的多样性
        if is_relight:
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
        face = cv2.resize(face, (size, size))

        cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)

        index += 1

        # cv2.imshow('image', face)

    return index


def get_my_face():
    logger.info('Getting "My Face"...')
    index = 1
    while True:
        if index <= 1000:
            # 从摄像头读取照片
            success, img = camera.read()
            index = img_handle(img, index, True, my_faces_dir)

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break

    logger.info('Get "My Face" Finished!')


def get_other_face():
    logger.info('Getting "Other Face"...')
    index = 1
    for (path, dirnames, filenames) in os.walk(other_faces_input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                img_path = path + '/' + filename
                # 从文件读取图片
                img = cv2.imread(img_path)
                index = img_handle(img, index, False, other_faces_output_dir)

                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    logger.info('Get "Other Face" Finished!')
                    sys.exit(0)


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()  # dlib的特征提取器
    camera = cv2.VideoCapture(0)

    # config read
    my_faces_dir = CONFIG.get("my_faces_dir")
    other_faces_input_dir = CONFIG.get("other_faces_input_dir")
    other_faces_output_dir = CONFIG.get("other_faces_input_dir")
    size = CONFIG.get("size")

    # check Dir
    os.makedirs(my_faces_dir) if not os.path.exists(my_faces_dir) else None
    if not os.path.exists(other_faces_input_dir):
        logger.error(f"directory name not found '{other_faces_output_dir}'")
        exit(-1)
    os.makedirs(other_faces_output_dir) if not os.path.exists(other_faces_output_dir) else None

    get_my_face()
