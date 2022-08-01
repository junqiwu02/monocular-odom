import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot


# Create noise models
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0004, 0.0004, 0.002]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0, 0, 0]))
PLOT_STEP = 50 # poses to skip when plotting


gt = np.loadtxt('data/kitti/00.txt').reshape((-1, 3, 4))
pred = np.loadtxt('data/kitti/00_pred_final.txt').reshape((-1, 3, 4))

pos_t = gt[:,:,3] # extract position vectors from pose mat
pos_p = pred[:,:,3]

rot_p = pred[:,:,:3]

plt.plot(pos_t[:,0], pos_t[:,2], label='Truth')
plt.plot(pos_p[:,0], pos_p[:,2], label='Pred')
plt.legend()

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

prev_pos = pos_p[0]
prev_rot = rot_p[0]
# add the prior pose
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(prev_pos[0], prev_pos[2], 0), PRIOR_NOISE))
initial.insert(0, gtsam.Pose2(prev_pos[0], prev_pos[2], 0))

for i, (p, r) in enumerate(zip(pos_p[1:], rot_p[1:])):
    i += 1 # reindex
    delta_pos = p - prev_pos
    delta_rot = r @ prev_rot.T

    # add factor for pose transition and initial pose estimate for later optimization
    # Pose2 translations must be relative to the CURRENT heading, not relative to the world frame
    graph.add(gtsam.BetweenFactorPose2(i - 1, i, gtsam.Pose2(delta_pos[0], delta_pos[2], 0), ODOMETRY_NOISE))
    initial.insert(i, gtsam.Pose2(p[0], p[2], 0))

    prev_pos = p
    prev_rot = r

# hardcoded loop closures
# delta_rot = gtsam.Rot3(rot_p[1412] @ rot_p[573].T).yaw()
graph.add(gtsam.BetweenFactorPose2(1412, 573, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))
# delta_rot = gtsam.Rot3(rot_p[1644] @ rot_p[207].T).yaw()
graph.add(gtsam.BetweenFactorPose2(1644, 207, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(2476, 424, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))
# delta_rot = gtsam.Rot3(rot_p[3429] @ rot_p[2476].T).yaw()
graph.add(gtsam.BetweenFactorPose2(3429, 2476, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))
# delta_rot = gtsam.Rot3(rot_p[3681] @ rot_p[745].T).yaw()
graph.add(gtsam.BetweenFactorPose2(3681, 745, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))
graph.add(gtsam.BetweenFactorPose2(4463, 6, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))

# graph.add(gtsam.BetweenFactorPose2(1412, 573, gtsam.Pose2(pos_p[1412][0] - pos_p[573][0], pos_p[1412][2] - pos_p[573][2], 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(1644, 207, gtsam.Pose2(pos_p[1644][0] - pos_p[207][0], pos_p[1644][2] - pos_p[207][2], 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(2476, 424, gtsam.Pose2(pos_p[2476][0] - pos_p[424][0], pos_p[2476][2] - pos_p[424][2], 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(3429, 2476, gtsam.Pose2(pos_p[3429][0] - pos_p[2476][0], pos_p[3429][2] - pos_p[2476][2], 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(3681, 745, gtsam.Pose2(pos_p[3681][0] - pos_p[745][0], pos_p[3681][2] - pos_p[745][2], 0), ODOMETRY_NOISE))
# graph.add(gtsam.BetweenFactorPose2(4463, 6, gtsam.Pose2(pos_p[4463][0] - pos_p[6][0], pos_p[4463][2] - pos_p[6][2], 0), ODOMETRY_NOISE))

# optimize
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

marginals = gtsam.Marginals(graph, result)

for i in range(0, pos_p.shape[0], PLOT_STEP):
    gtsam_plot.plot_pose2(plt.gcf().number, result.atPose2(i), 10, marginals.marginalCovariance(i))

plt.show()