import numpy as np
import sys

DIST_THRESH = 5 # distance threshold in meters for a loop
CHECK_INTERVAL = 5 # number of frames to skip in between checks for loop
LOOP_INTERNVAL = 200 # minimum frames between loop connection

if len(sys.argv) < 2:
    sys.exit('Please enter a sequence id and try again!')

seq_id = sys.argv[1]

truth = np.loadtxt(f'data/kitti/poses/{seq_id}.txt').reshape((-1, 3, 4)) # ground truth poses
pos_t = truth[:,:,3] # extract position vectors from pose mat

loops = []
i = 0
while i < pos_t.shape[0]:
    j = i + LOOP_INTERNVAL
    detected = False
    while j < pos_t.shape[0]: # search through all other pairs
        if np.linalg.norm(pos_t[i] - pos_t[j]) < DIST_THRESH:
            loops.append([i, j])
            j += LOOP_INTERNVAL # skip ahead
            detected = True
        else:
            j += CHECK_INTERVAL
    i += LOOP_INTERNVAL if detected else CHECK_INTERVAL

print(f'Found loops: {loops}')
with open(f'results/loops/{seq_id}_loops.txt', 'w') as f:
    for i, j in loops:
        f.write(f'{i} {j}\n')