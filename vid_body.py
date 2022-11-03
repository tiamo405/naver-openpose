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

# def createjson(poses,opt):
#     data_dict={
#         "version":1.3,
#         "people":[{"person_id":[-1],"pose_keypoints_2d":list(np.array(poses).reshape(75))}]
#     }
#     data_string = json.dumps(data_dict)

#     myjsonfile = open("VITON-HD/datasets/test/openpose-json/10550_00_keypoints.json", "w")
#     myjsonfile.write(data_string)
#     myjsonfile.close()
#

tp = torch_openpose.torch_openpose('body_25')
cap = cv2.VideoCapture("openpose_body_25/images/Messenger.mp4")
i= 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
result = cv2.VideoWriter('1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (frame_width, frame_height))
while(cap.isOpened()):
     
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        try :
            poses =tp(frame)
            frame = util.draw_bodypose(frame, poses,'body_25')  
            # cv2.imwrite('openpose_body_25/vid/'+str(i)+'.jpg', frame)
        except :
            i+=1
        result.write(frame)
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else :
        break
    
# When everything done, release
# the video capture object
cap.release()
result.release()

# cv2.destroyAllWindows()


