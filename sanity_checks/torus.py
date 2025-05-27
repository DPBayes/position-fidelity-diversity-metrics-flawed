import numpy as np 
from scipy import stats
from scipy.spatial.transform import Rotation


def sample_unit_hypersphere_single(dim):
    while True:
        attempt = np.random.uniform(-1, 1, dim)
        if np.sum(attempt**2) <= 1:
            return attempt

def sample_hypersphere(size, dim, center, radius):
    result = np.zeros((size, dim))
    for i in range(size):
        result[i, :] = sample_unit_hypersphere_single(dim)

    result *= radius 
    result += center
    return result

def sample_torus(size):
    angles = np.random.uniform(0, 2 * np.pi, size)
    circle_points = sample_hypersphere(size, 2, np.array((1, 0)), 0.1)
    circle_points_3d = np.zeros((size, 3))
    circle_points_3d[:, 0] = circle_points[:, 0]
    circle_points_3d[:, 2] = circle_points[:, 1]

    rotations = Rotation.from_euler("z", angles)
    rotated = rotations.apply(circle_points_3d)
    return rotated


def run_sphere_torus(metric_computer_factory):
    n_real = 1000
    n_syn_list = np.logspace(2, 4, 20).astype(int) # 100 to 10000

    results = {}

    for real_dist in ["real_sphere", "real_torus"]:
        results[real_dist] = {}

        if real_dist == "real_sphere":
            real = sample_hypersphere(n_real, 3, 0, 0.8)
        else:
            real = sample_torus(n_real)
        metric_computer = metric_computer_factory(real)

        for n_syn in n_syn_list:
            if real_dist == "real_sphere":
                syn = sample_torus(n_syn)
            else:
                syn = sample_hypersphere(n_syn, 3, 0, 0.8)
            metrics = metric_computer.compute_metric(syn)
            results[real_dist][n_syn] = metrics

    return results
