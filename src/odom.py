import cv2
import numpy as np

MIN_KP = 500
MAX_KP = 1000


img0 = None
p0 = []
pos = None
rot = None

fast = cv2.FastFeatureDetector_create()
cap = cv2.VideoCapture('./data/drive.mp4')

traj = np.zeros((480, 640, 3), dtype=np.uint8)
traj[:] = (255, 255, 255)


while True:
    if img0 is None:
        ret, img0 = cap.read()
    if len(p0) < MIN_KP:
        p0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST corner detection
    if len(p0) < MIN_KP: # still low number of keypoints, continue to next frame
        ret, img0 = cap.read()
        continue
    if len(p0) > MAX_KP:
        np.random.shuffle(p0) # limit max num of keypoints to save processing time
        p0 = p0[:MAX_KP]

    ret, img1 = cap.read()
    p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, nextPts=None) # Use KLT optical flow to track keypoints
    mask = status.flatten() == 1 # keep only points which were successfully tracked 
    p0 = p0[mask,:]
    p1 = p1[mask,:]

    E, inliers = cv2.findEssentialMat(p1, p0, focal=400, pp=(427, 240), method=cv2.RANSAC) # calc the Essential matrix which transforms between camera poses
    # E solves the inner product aTEb = 0

    retval, R, t, inliers = cv2.recoverPose(E, p1, p0, focal=400, pp=(427, 240), mask=inliers) # calc rotation and translation from E
    # NOTE R and t returned will always be unit length since scale cannot be inferred from monocular vision alone
    # Therefore they must be scaled using another source such as a speedometer, otherwise motion while stationary can seem to vary greatly

    scale = np.average(np.linalg.norm(p0 - p1, axis=1)) # since we have no speedometer, estimate scale as the magnitude of optical flow between frames
    print(int(scale))

    if pos is None:
        pos = t
        rot = R
    elif scale > 2 and t[2] > t[0] and t[2] > t[1]: # assume movement is dominantly foward to avoid issues with moving objects and ignore small movements for less noise when stopped
        # update position and rotation
        pos += scale * rot @ t # translation is calc'd first because t is relative to original heading
        rot = R @ rot

    def graph_coords(v): # convert position vector to graph image coords
        return (int(v[0] / 10) + traj.shape[1] // 2, int(v[2] / 10) + traj.shape[0] // 2)

    # visualize
    cv2.imshow("img1 (Press 'q' to quit)", cv2.drawKeypoints(img1, cv2.KeyPoint_convert(p1), outImage=None, color=(255,0,0)))
    cv2.circle(traj, graph_coords(pos), 1, (255,127,127), 2)
    curr = traj.copy()
    tail = pos + rot @ np.array([0, 0, -200]).reshape((3, 1)) # tail of the arrow will be 1000 units behind the head
    cv2.arrowedLine(curr, graph_coords(tail), graph_coords(pos), (255,0,0), 2, tipLength=0.333)
    cv2.imshow("trajectory", curr)
    if cv2.waitKey(1) == ord('q'):
        break
    
    img0 = img1 # go next
    p0 = p1

cap.release()
cv2.destroyAllWindows()