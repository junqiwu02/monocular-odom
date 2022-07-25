import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot


# Create noise models
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0002, 0.0002, 0.001]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0, 0, 0]))
PLOT_STEP = 50 # poses to skip when plotting


gt = np.loadtxt('data/kitti/00.txt').reshape((-1, 3, 4))
pred = np.loadtxt('data/kitti/00_pred_final.txt').reshape((-1, 3, 4))

pos_t = gt[:,:,3] # extract position vectors from pose mat
pos_p = pred[:,:,3]

plt.plot(pos_t[:,0], pos_t[:,2], label='Truth')
plt.plot(pos_p[:,0], pos_p[:,2], label='Pred')
plt.legend()

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

prev = pos_p[0]
# add the prior pose
graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(prev[0], prev[2], 0.0), PRIOR_NOISE))
initial.insert(1, gtsam.Pose2(prev[0], prev[2], 0.0))

for i, p in enumerate(pos_p[1:]):
    i += 2 # reindex since gtsam indices start at 1
    delta = p - prev

    # add factor for pose transition and initial pose estimate for later optimization
    graph.add(gtsam.BetweenFactorPose2(i - 1, i, gtsam.Pose2(delta[0], delta[2], 0.0), ODOMETRY_NOISE))
    initial.insert(i, gtsam.Pose2(p[0], p[2], 0.0))

    prev = p

# optimize
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

marginals = gtsam.Marginals(graph, result)

for i in range(1, pos_p.shape[0], PLOT_STEP):
    gtsam_plot.plot_pose2(plt.gcf().number, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

plt.show()