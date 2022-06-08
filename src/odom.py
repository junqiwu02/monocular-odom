import cv2
import glob

PATH = './data/deer_walk/cam0/data'
REDETECT_THRESH = 500

fast = cv2.FastFeatureDetector_create()

img0 = None
p0 = []

for f in glob.glob(f'{PATH}/*.png'):
    if img0 is None:
        img0 = cv2.imread(f)
    if len(p0) < REDETECT_THRESH:
        p0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST detection

    img1 = cv2.imread(f)
    p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, nextPts=None)
    p1 = p1[status.flatten()==1,:] # keep only found points 
    print(f'{len(p1)} keypoints tracked!')

    cv2.imshow('keypoints', cv2.drawKeypoints(img1, cv2.KeyPoint_convert(p1), outImage=None, color=(255,0,0)))
    k = cv2.waitKey()
    if k == ord('q'):
        break

    img0 = img1 # go next
    p0 = p1

