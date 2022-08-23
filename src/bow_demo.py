import cv2

bow = cv2.BOWImgDescriptorExtractor(cv2.ORB_create(), cv2.DescriptorMatcher_create('BruteForce'))
fast = cv2.FastFeatureDetector_create()

img0 = cv2.imread('data/kitti/image_0/000000.png')
kp = fast.detect(img0, mask=None)

out = bow.compute(img0, kp)
print(out.shape)