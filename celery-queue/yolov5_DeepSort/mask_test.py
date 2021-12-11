# -*- coding: utf-8 -*-
# @TIME : 2021/5/14 11:20
# @AUTHOR : Xu Bai
# @FILE : mask_test.py
# @DESCRIPTION :
import cv2
import matplotlib.pyplot as plt
import numpy as np

mask_threth = 50

img = cv2.imread('mask_o.jpg')  # 自己qq截图一张图片就行，要大于下面的坐标点

# binary mask
coordinates = []
coordinate1 = [[[40, 135], [168, 132], [164, 330], [2, 328]]]
coordinate2 = [[[300, 300], [600, 300], [600, 600], [300, 600]]]
# 顺时针顺序
coordinate3 = [[[700,190], [720, 880],  [1900, 860],[1900, 170] ,]]
coordinate1 = np.array(coordinate1)
coordinate2 = np.array(coordinate2)
coordinate3 = np.array(coordinate3)
# coordinates.append(coordinate1)
# coordinates.append(coordinate2)
coordinates.append(coordinate3)
mask = np.zeros(img.shape[:2], dtype=np.int8)
mask = cv2.fillPoly(mask, coordinates, 255)
cv2.imwrite('mask1.png', mask)

bbox_mask = mask
color_mask = np.array([0, 0, 0], dtype=np.uint8)
bbox_mask = bbox_mask.astype(np.bool)
# cv2.imwrite('./bbox_mask.png', bbox_mask)

img[bbox_mask] = img[bbox_mask] > color_mask
img = img[:, :, ::-1]
plt.imshow(img)
# plt.savefig('./result.png')
plt.show()
