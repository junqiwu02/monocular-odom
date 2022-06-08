import cv2
import numpy as np

MIN_KP = 1000
MAX_KP = 4000
SCALE = 1


img0 = None
p0 = []
pos = None
rot = None

fast = cv2.FastFeatureDetector_create()
cap = cv2.VideoCapture('./data/drive.mp4')

traj = np.zeros((720, 1280, 3), dtype=np.uint8)
traj[:] = (255, 255, 255)


while True:
    if img0 is None:
        ret, img0 = cap.read()
    if len(p0) < MIN_KP:
        p0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST corner detection
    if len(p0) < MIN_KP: # still low number of keypoints, continue to next frame
        ret, img0 = cap.read()
        continue

    ret, img1 = cap.read()
    p0 = p0[:MAX_KP] # limit max num of keypoints to save processing time
    p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, nextPts=None) # Use KLT optical flow to track keypoints
    mask = status.flatten() == 1 # keep only points which were successfully tracked 
    p0 = p0[mask,:]
    p1 = p1[mask,:]
    print(f'{len(p1)} keypoints tracked!')

    E, inliers = cv2.findEssentialMat(p0, p1, method=cv2.RANSAC, prob=0.999, threshold=1.0) # calc the Essential matrix which transforms between camera poses
    # E solves the inner product aTEb = 0

    retval, R, t, inliers = cv2.recoverPose(E, p0, p1) # calc rotation and translation from E
    if pos is None:
        pos = t
        rot = R
    else:
        # update position and rotation
        pos += SCALE * rot @ t # translation is calc'd first because t is relative to original heading
        rot = R @ rot

    cv2.imshow("img1 (Press 'q' to quit)", cv2.drawKeypoints(img1, cv2.KeyPoint_convert(p1), outImage=None, color=(255,0,0)))
    cv2.circle(traj, (int(pos[1]) + traj.shape[1] // 2, int(pos[2]) + traj.shape[0] // 2), 1, (255,0,0), 2)
    cv2.imshow("traj", traj)
    if cv2.waitKey(1) == ord('q'):
        break
    
    img0 = img1 # go next
    p0 = p1

cap.release()
cv2.destroyAllWindows()