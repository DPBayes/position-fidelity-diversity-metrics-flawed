import numpy as np 
from scipy import stats


def sample_uniform_hypersphere(size, dim, radius):
    gaussian_sample = stats.norm.rvs(size=(size, dim))
    norms = np.sqrt(np.sum(gaussian_sample**2, axis=1)).reshape((-1, 1))
    data = gaussian_sample / norms * radius
    return data


def run_uniform_hypersphere_surface(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    dim_list = [2, 16, 128]

    real_radius = 1
    syn_radii = np.linspace(0.1, 1.9, 51)

    results = {}
    for dim in dim_list:
        results[dim] = {}
        real = sample_uniform_hypersphere(n_real, dim, real_radius)
        metric_computer = metric_computer_factory(real)
        for syn_radius in syn_radii:
            syn = sample_uniform_hypersphere(n_syn, dim, syn_radius)
            metrics = metric_computer.compute_metric(syn)
            results[dim][syn_radius] = metrics

    return results

