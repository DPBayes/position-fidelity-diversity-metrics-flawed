import numpy as np 
from scipy import stats
from scipy.spatial.transform import Rotation


def sample_uniform_hypercube(size, dim, start, end):
    return np.random.uniform(start, end, (size, dim))


def run_uniform_hypercube_varying_size(metric_computer_factory):
    n_list = np.logspace(2, 4, 20).astype(int) # 100 to 10000
    dim_list = [1, 8, 64]
    overlap_volume = 0.2

    results = {}
    for dim in dim_list:
        results[dim] = {}
        for n in n_list:
            real = sample_uniform_hypercube(n, dim, 0, 1)
            metric_computer = metric_computer_factory(real)

            # (1 - syn_start)**d = overlap_volume
            # syn_start = 1 - overlap_volume**(1 / d)
            syn_start = 1 - overlap_volume**(1 / dim)
            syn = sample_uniform_hypercube(n, dim, syn_start, syn_start + 1)
            metrics = metric_computer.compute_metric(syn)
            results[dim][n] = metrics

    return results


def run_uniform_hypercube_varying_syn_size(metric_computer_factory):
    n_real = 1000
    n_syn_list = np.logspace(2, 4, 20).astype(int) # 100 to 10000
    dim_list = [1, 8, 64]
    overlap_volume = 0.2

    results = {}
    for dim in dim_list:
        results[dim] = {}
        real = sample_uniform_hypercube(n_real, dim, 0, 1)
        metric_computer = metric_computer_factory(real)
        for n_syn in n_syn_list:
            # (1 - syn_start)**d = overlap_volume
            # syn_start = 1 - overlap_volume**(1 / d)
            syn_start = 1 - overlap_volume**(1 / dim)
            syn = sample_uniform_hypercube(n_syn, dim, syn_start, syn_start + 1)
            metrics = metric_computer.compute_metric(syn)
            results[dim][n_syn] = metrics

    return results

