'''
https://github.com/Hzzone/pytorch-openpose
'''
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('openpose_body_25/model/body_pose_model.pth')
hand_estimation = Hand('openpose_body_25/model/hand_pose_model.pth')

test_image = 'VITON-HD/datasets/test/image/10550_00.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
overlay_image = np.ones(oriImg.shape, np.uint8)  * 0
# overlay_image = util.draw_bodypose(overlay_image, candidate, subset)
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
