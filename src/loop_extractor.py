import numpy as np
import sys
from collections import defaultdict

DIST_THRESH = 5 # distance threshold in meters for a loop
CHECK_INTERVAL = 5 # number of frames to skip in between checks for loop
LOOP_INTERNVAL = 200 # minimum frames between loop connection

if len(sys.argv) < 2:
    sys.exit('Please enter a sequence id and try again!')

seq_id = sys.argv[1]

truth = np.loadtxt(f'data/kitti/poses/{seq_id}.txt').reshape((-1, 3, 4)) # ground truth poses
pos_t = truth[:,:,3] # extract position vectors from pose mat

loops = defaultdict(lambda: [])
i = 0
while i < pos_t.shape[0]:
    j = i + LOOP_INTERNVAL
    while j < pos_t.shape[0]:
        if np.linalg.norm(pos_t[i] - pos_t[j]) < DIST_THRESH:
            loops[i].append(j)
            j += LOOP_INTERNVAL # skip ahead
        else:
            j += CHECK_INTERVAL
    if i in loops:
        i += LOOP_INTERNVAL
    else:
        i += CHECK_INTERVAL

print(f'Found loops: {loops}')
with open(f'results/{seq_id}_loops.txt', 'w') as f:
    for start, ends in loops.items():
        for end in ends:
            f.write(f'{start} {end}\n')