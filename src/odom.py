import numpy as np
import cv2

PATH = './data/deer_walk/cam0/data'

fast = cv2.FastFeatureDetector_create()

img0 = cv2.imread(f'{PATH}/0000000000000000000.png')

kp = fast.detect(img0, mask=None)
print(f'{len(kp)} keypoints found')
img0 = cv2.drawKeypoints(img0, kp, outImage=None, color=(255,0,0))

cv2.imshow('img', img0)
cv2.waitKey()
