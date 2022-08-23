import cv2
import numpy as np
import glob
import util
import gtsam
import math
import sys

MIN_KP = 1000
MAX_KP = 2000

FOCAL = 718.8560
PP = (607.1928, 185.2157)

if len(sys.argv) < 2:
    sys.exit('Please enter a sequence id and try again!')

seq_id = sys.argv[1]

img0 = None
kp0 = []
pos = np.zeros((3,1), dtype=np.float64)
rot = np.eye(3,3)

pos_pt = np.zeros((3,1), dtype=np.float64) # prev ground truth pos
rot_pt = np.eye(3,3)

fast = cv2.FastFeatureDetector_create()

# trajectory visualization
traj = np.zeros((480, 640, 3), dtype=np.uint8)
traj[:] = (255, 255, 255)

def graph_coords(v): # convert position vector to graph image coords
    return (int(v[0] / 2) + traj.shape[1] // 2, -int(v[2] / 2) + traj.shape[0] * 7 // 8)

def draw_arrow(img, t, R, color):
    tail = t + R @ np.array([0, 0, -30]).reshape((3, 1)) # tail of the arrow will be 30 units behind the head
    cv2.arrowedLine(img, graph_coords(tail), graph_coords(t), color, 2, tipLength=0.333)

imgs = sorted(glob.glob(f'data/kitti/sequences/{seq_id}/image_0/*.png'))
truth = np.loadtxt(f'data/kitti/poses/{seq_id}.txt').reshape((-1, 3, 4)) # ground truth poses
pred = [] # predicted output

for f, pose_t in zip(util.progress_bar(imgs, 'Progress: '), truth):
    pos_t = pose_t[:,3].reshape((3,1)) # split ground truth pose into pos vector and rot matrix
    rot_t = pose_t[:,:3]

    img1 = cv2.imread(f)
    kp1 = []

    if img0 is not None and len(kp0) < MIN_KP:
        kp0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST corner detection
        
    if len(kp0) >= MIN_KP: # ensure enough keypoints and run odom
        if len(kp0) > MAX_KP:
            np.random.shuffle(kp0) # limit max num of keypoints to save processing time
            kp0 = kp0[:MAX_KP]

        kp1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, kp0, nextPts=None) # Use KLT optical flow to track keypoints
        mask = status.flatten() == 1 # keep only points which were successfully tracked 
        kp0 = kp0[mask,:]
        kp1 = kp1[mask,:]

        E, inliers = cv2.findEssentialMat(kp1, kp0, focal=FOCAL, pp=PP, method=cv2.RANSAC) # calc the Essential matrix which transforms between camera poses
        # E solves the inner product aTEb = 0

        retval, R, t, inliers = cv2.recoverPose(E, kp1, kp0, focal=FOCAL, pp=PP, mask=inliers) # calc rotation and translation from E
        # NOTE R and t returned will always be unit length since scale cannot be inferred from monocular vision alone
        # Therefore they must be scaled using another source such as a speedometer, otherwise motion while stationary can seem to vary greatly

        scale = np.linalg.norm(pos_t - pos_pt)

        if scale > 0.1 and t[2] > t[0] and t[2] > t[1]: # assume movement is dominantly foward to avoid issues with moving objects and ignore small movements for less noise when stopped
            # update position and rotation
            # NOTE : it seems that bumps in the road lead to large y translation errors
            # Temp heuristic solution will just to be to project the vector onto the x-z plane
            dt = rot @ t
            dt[1] = 0
            dt /= np.linalg.norm(dt)
            pos += scale * dt # translation is calc'd first because t is relative to original heading
            rot = R @ rot

    # visualize
    cv2.imshow("img1 (Press 'q' to quit)", cv2.drawKeypoints(img1, cv2.KeyPoint_convert(kp1), outImage=None, color=(255,0,0)))
    cv2.circle(traj, graph_coords(pos), 1, (255,127,127), 2)
    cv2.circle(traj, graph_coords(pos_t), 1, (127,255,127), 2)
    graph = traj.copy()
    draw_arrow(graph, pos, rot, (255, 0, 0))
    draw_arrow(graph, pos_t, rot_t, (0, 127, 0))
    cv2.putText(graph, f'Translation error: {(pos - pos_t).flatten()}m', (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.putText(graph, f'Rotation error: {gtsam.Rot3(rot @ rot_t.T).xyz() * 180 / math.pi}deg', (0, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("trajectory", graph)
    if cv2.waitKey(1) == ord('q'):
        break
    
    img0 = img1 # go next
    kp0 = kp1

    pos_pt = pos_t
    rot_pt = rot_t

    pose_p = np.zeros((3,4)) # output predicted pose
    pose_p[:,3] = pos.flatten()
    pose_p[:,:3] = rot
    pred.append(pose_p.flatten())

np.savetxt(f'results/{seq_id}_pred.txt', pred)
cv2.destroyAllWindows()