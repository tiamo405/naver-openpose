#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:17:47 2020

@author: joe
github pytorch_openpose_body_25
"""

import argparse
from typing import overload
from src import torch_openpose,util
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import copy
from src import model
from src.body import Body
from src.hand import Hand
from os import path as osp

def createjson(poses,opt):
    data_dict={
        "version":1.3,
        "people":[{"person_id":[-1],"pose_keypoints_2d":list(np.array(poses).reshape(75))}]
    }
    data_string = json.dumps(data_dict)

    myjsonfile = open("VITON-HD/datasets/test/openpose-json/10550_00_keypoints.json", "w")
    myjsonfile.write(data_string)
    myjsonfile.close()
#----------------------------
def draw_hand(img,opt) :
    oriImg = img.copy()  # B,G,R order
    candidate, subset = body_estimation(oriImg)

    overlay_image = np.ones(oriImg.shape, np.uint8)  * 0

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        
        all_hand_peaks.append(peaks)
    overlay_image = cv2.imread("VITON-HD/datasets/test/openpose-img/10550_00_rendered.png")
    overlay_image = util.draw_handpose(overlay_image, all_hand_peaks)
    overlay_image = cv2.resize(overlay_image, (768,1024))

    cv2.imwrite("VITON-HD/datasets/test/openpose-img/10550_00_rendered.png", overlay_image)
    # cv2.imwrite("tet.png", overlay)
    print(overlay_image.shape)
# ----------------------------
def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='VITON-HD/datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='VITON-HD/checkpoints/')
    parser.add_argument('--save_dir', type=str, default='results/')

    opt = parser.parse_args()
    return opt
if __name__ == "__main__":
    opt = get_opt()
    body_estimation = Body('openpose_body_25/model/body_pose_model.pth')
    hand_estimation = Hand('openpose_body_25/model/hand_pose_model.pth')
    img_names = []
    c_names = []
    with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
        for line in f.readlines():
            img_name, c_name = line.strip().split()
            img_names.append(img_name)
            c_names.append(c_name)
    tp = torch_openpose.torch_openpose('body_25')
    img = cv2.imread("VITON-HD/datasets/test/image/10550_00.jpg")
    poses = tp(img)
    createjson(poses,opt)
    overlay_image = np.ones(img.shape, np.uint8)  * 0
    overlay_image = cv2.resize(overlay_image, (768,1024))
    img_body = util.draw_bodypose(overlay_image, poses,'body_25')
    cv2.imwrite("VITON-HD/datasets/test/openpose-img/10550_00_rendered.png",img_body)
    draw_hand(img, opt)
    print("done")
