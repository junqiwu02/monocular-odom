import cv2
import numpy as np
import glob
import util

MIN_KP = 500
MAX_KP = 1000

FOCAL = 718.8560
PP = (607.1928, 185.2157)


img0 = None
p0 = []
pos = np.zeros((3,1), dtype=np.float64)
rot = np.eye(3,3)

pos_pt = np.zeros((3,1), dtype=np.float64) # prev ground truth pos
rot_pt = np.eye(3,3)

fast = cv2.FastFeatureDetector_create()

traj = np.zeros((480, 640, 3), dtype=np.uint8)
traj[:] = (255, 255, 255)

gt = open('data/kitti/00.txt') # ground truth poses
pred = open('data/kitti/00_pred.txt', 'w') # predicted output

for f in util.progress_bar(sorted(glob.glob('data/kitti/image_0/*.png')), 'Progress: '):
    pose_t = np.array(gt.readline().split(' '), dtype=np.float64).reshape((3,4)) # read ground truth pose
    pos_t = pose_t[:,3].reshape((3,1)) # split pose into pos vector and rot matrix
    rot_t = pose_t[:,:3]

    img1 = cv2.imread(f)
    p1 = []

    if img0 is not None and len(p0) < MIN_KP:
        p0 = cv2.KeyPoint_convert(fast.detect(img0, mask=None)) # low num of keypoints, run FAST corner detection
        
    if len(p0) >= MIN_KP: # ensure enough keypoints and run odom
        if len(p0) > MAX_KP:
            np.random.shuffle(p0) # limit max num of keypoints to save processing time
            p0 = p0[:MAX_KP]

        p1, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, nextPts=None) # Use KLT optical flow to track keypoints
        mask = status.flatten() == 1 # keep only points which were successfully tracked 
        p0 = p0[mask,:]
        p1 = p1[mask,:]

        E, inliers = cv2.findEssentialMat(p1, p0, focal=FOCAL, pp=PP, method=cv2.RANSAC) # calc the Essential matrix which transforms between camera poses
        # E solves the inner product aTEb = 0

        retval, R, t, inliers = cv2.recoverPose(E, p1, p0, focal=FOCAL, pp=PP, mask=inliers) # calc rotation and translation from E
        # NOTE R and t returned will always be unit length since scale cannot be inferred from monocular vision alone
        # Therefore they must be scaled using another source such as a speedometer, otherwise motion while stationary can seem to vary greatly

        scale = np.linalg.norm(pos_t - pos_pt)

        if scale > 0.1 and t[2] > t[0] and t[2] > t[1]: # assume movement is dominantly foward to avoid issues with moving objects and ignore small movements for less noise when stopped
            # update position and rotation
            pos += scale * rot @ t # translation is calc'd first because t is relative to original heading
            rot = R @ rot

    def graph_coords(v): # convert position vector to graph image coords
        return (int(v[0] / 2) + traj.shape[1] // 2, -int(v[2] / 2) + traj.shape[0] * 7 // 8)

    def draw_arrow(img, t, R, color):
        tail = t + R @ np.array([0, 0, -30]).reshape((3, 1)) # tail of the arrow will be 30 units behind the head
        cv2.arrowedLine(img, graph_coords(tail), graph_coords(t), color, 2, tipLength=0.333)

    # visualize
    cv2.imshow("img1 (Press 'q' to quit)", cv2.drawKeypoints(img1, cv2.KeyPoint_convert(p1), outImage=None, color=(255,0,0)))
    cv2.circle(traj, graph_coords(pos), 1, (255,127,127), 2)
    cv2.circle(traj, graph_coords(pos_t), 1, (127,255,127), 2)
    graph = traj.copy()
    draw_arrow(graph, pos, rot, (255, 0, 0))
    draw_arrow(graph, pos_t, rot_t, (0, 127, 0))
    cv2.putText(graph, f'Translation error: {abs(pos - pos_t)}m', (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("trajectory", graph)
    if cv2.waitKey(1) == ord('q'):
        break
    
    img0 = img1 # go next
    p0 = p1

    pos_pt = pos_t
    rot_pt = rot_t

    pose_p = np.zeros((3,4)) # output predicted pose
    pose_p[:,3] = pos.flatten()
    pose_p[:,:3] = rot
    pred.write(f"{' '.join(pose_p.flatten().astype(str))}\n")

gt.close()
pred.close()
cv2.destroyAllWindows()