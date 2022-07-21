from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

# first use mpl or whatever library the gtsam examples use to plot both pred and gt trajectories
gt = np.loadtxt('data/kitti/00.txt').reshape((-1, 3, 4))
pred = np.loadtxt('data/kitti/00_pred_final.txt').reshape((-1, 3, 4))

pos_t = gt[:,:,3] # extract position vectors from pose mat
pos_p = pred[:,:,3]

plt.plot(pos_t[:,0], pos_t[:,2], label='Truth')
plt.plot(pos_p[:,0], pos_p[:,2], label='Pred')
plt.legend()
plt.show()