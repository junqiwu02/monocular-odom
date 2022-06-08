import cv2
import numpy as np
import glob

PATH = './data/deer_walk/cam0/data'
REDETECT_THRESH = 500

fast = cv2.FastFeatureDetector_create()

img0 = None
p0 = []
traj = np.zeros((600, 600, 3), dtype=np.uint8)
traj[:] = (255, 255, 255)
pos = None
rot = None

for f in glob.glob(f'{PATH}/*.png'):
    if img0 is None:
        img0 = cv2.imread(f)
    if len(p0) < REDETECT_THRESH:
        p0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST corner detection

    img1 = cv2.imread(f)
    p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, nextPts=None) # Use KLT optical flow to track keypoints
    mask = status.flatten() == 1 # keep only points which were successfully tracked 
    p0 = p0[mask,:]
    p1 = p1[mask,:]
    print(f'{len(p1)} keypoints tracked!')

    E, inliers = cv2.findEssentialMat(p0, p1, method=cv2.RANSAC) # calc the Essential matrix which transforms between camera poses
    # E solves the inner product aTEb = 0

    retval, R, t, inliers = cv2.recoverPose(E, p0, p1) # calc rotation and translation from E
    if pos is None:
        pos = t
        rot = R
    # update position and rotation
    pos += rot @ t # translation is calc'd first because t is relative to original heading
    rot = R @ rot

    cv2.imshow("img1 (Press 'q' to quit)", cv2.drawKeypoints(img1, cv2.KeyPoint_convert(p1), outImage=None, color=(255,0,0)))
    cv2.circle(traj, (int(pos[0]) + 300, int(pos[2]) + 300), 1, (255,0,0), 2)
    cv2.imshow("traj", traj)
    if cv2.waitKey() == ord('q'):
        break
    
    img0 = img1 # go next
    p0 = p1

