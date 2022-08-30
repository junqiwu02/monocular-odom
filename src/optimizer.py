import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot
import sys


# Create noise models
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0004, 0.0004, 0.002]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0, 0, 0]))
PLOT_STEP = 50 # poses to skip when plotting

if len(sys.argv) < 2:
    sys.exit('Please enter a sequence id and try again!')

seq_id = sys.argv[1]


truth = np.loadtxt(f'data/kitti/poses/{seq_id}.txt').reshape((-1, 3, 4))
pred = np.loadtxt(f'results/poses/{seq_id}_pred.txt').reshape((-1, 3, 4))
loops = np.loadtxt(f'results/loops/{seq_id}_loops.txt', dtype=int).reshape((-1, 2))

pos_t = truth[:,:,3] # extract position vectors from pose mat
pos_p = pred[:,:,3]

rot_p = pred[:,:,:3]

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

# add loop closures
for i, j in loops:
    graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(0, 0, 0), ODOMETRY_NOISE))

# optimize
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

marginals = gtsam.Marginals(graph, result)

def values_to_array(values):
    return np.array([[values.atPose2(i).x(), 0, values.atPose2(i).y()] for i in range(values.size())])

pos_opt = values_to_array(result)
    

plt.plot(pos_t[:,0], pos_t[:,2], label='Truth')
plt.plot(pos_p[:,0], pos_p[:,2], label='Pred', linestyle='dotted')
plt.plot(pos_opt[:,0], pos_opt[:,2], label='Optimized', linestyle='dashdot')
plt.scatter(pos_t[loops[:,0],0], pos_t[loops[:,0],2], label='Loops')
plt.legend()

if len(sys.argv) >= 3 and sys.argv[2] == '-c': # plot covariances
    for i in range(0, pos_p.shape[0], PLOT_STEP):
        gtsam_plot.plot_pose2(plt.gcf().number, result.atPose2(i), 10, marginals.marginalCovariance(i))

# output
times = np.loadtxt(f'data/kitti/sequences/{seq_id}/times.txt')

out_true = np.zeros((times.shape[0], 8))
out_pred = np.zeros((times.shape[0], 8))
out_opt = np.zeros((times.shape[0], 8))

out_true[:,0] = times
out_true[:,1] = pos_t[:,0]
out_true[:,3] = pos_t[:,2]
np.savetxt(f'results/eval/{seq_id}/true.txt', out_true)

out_pred[:,0] = times
out_pred[:,1] = pos_p[:,0]
out_pred[:,3] = pos_p[:,2]
np.savetxt(f'results/eval/{seq_id}/predicted.txt', out_pred)

out_opt[:,0] = times
out_opt[:,1] = pos_opt[:,0]
out_opt[:,3] = pos_opt[:,2]
np.savetxt(f'results/eval/{seq_id}/optimized.txt', out_opt)

plt.savefig(f'results/eval/{seq_id}/plot.png')
plt.show()