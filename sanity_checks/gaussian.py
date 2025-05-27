import numpy as np 
from scipy import stats


def run_gaussian_mean_difference(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    dim_list = [1, 8, 64]

    real_mean = 0
    real_std = 1
    syn_means_by_dim = {
        1: np.linspace(-6, 6, 51),
        8: np.linspace(-3, 3, 51),
        64: np.linspace(-1, 1, 51),
    }
    syn_std = real_std


    results = {}
    for dim in dim_list:
        results[dim] = {}
        real = stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, dim))
        metric_computer = metric_computer_factory(real)
        for syn_mean in syn_means_by_dim[dim]:
            syn = stats.norm.rvs(loc=syn_mean, scale=syn_std, size=(n_syn, dim))
            metrics = metric_computer.compute_metric(syn)
            results[dim][syn_mean] = metrics

    return results


def run_gaussian_mean_difference_with_outlier(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    dim_list = [1, 8, 64]

    real_mean = 0
    real_std = 1
    syn_means_by_dim = {
        1: np.linspace(-6, 6, 51),
        8: np.linspace(-3, 3, 51),
        64: np.linspace(-1, 1, 51),
    }
    syn_std = real_std


    results = {}
    for dim in dim_list:
        results[dim] = {}
        real = stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, dim))

        outlier_pos = np.max(syn_means_by_dim[dim])
        real_with_outlier = np.concatenate([real, outlier_pos * np.ones((1, dim))], axis=0)

        metric_computer = metric_computer_factory(real)
        metric_computer_real_outlier = metric_computer_factory(real_with_outlier)

        for syn_mean in syn_means_by_dim[dim]:
            syn = stats.norm.rvs(loc=syn_mean, scale=syn_std, size=(n_syn, dim))
            syn_with_outlier = np.concatenate([syn, outlier_pos * np.ones((1, dim))], axis=0)

            metrics_syn_outlier = metric_computer.compute_metric(syn_with_outlier)
            metrics_real_outlier = metric_computer_real_outlier.compute_metric(syn)
            results[dim][syn_mean] = {}
            results[dim][syn_mean]["real_outlier"] = metrics_real_outlier
            results[dim][syn_mean]["syn_outlier"] = metrics_syn_outlier

    return results


def run_gaussian_std_difference(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    dim_list = [1, 8, 64]

    real_mean = 0
    real_std = 1
    syn_mean = real_mean
    syn_stds_by_dim = {
        1: np.logspace(-3, 3, 51, base=10) * real_std,
        8: np.logspace(-1, 1, 51, base=10) * real_std,
        64: np.logspace(-0.5, 0.5, 51, base=10) * real_std,
    }


    results = {}
    for dim in dim_list:
        results[dim] = {}
        real = stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, dim))
        metric_computer = metric_computer_factory(real)
        for syn_std in syn_stds_by_dim[dim]:
            syn = stats.norm.rvs(loc=syn_mean, scale=syn_std, size=(n_syn, dim))
            metrics = metric_computer.compute_metric(syn)
            results[dim][syn_std] = metrics

    return results


def run_gaussian_scaling_one_dimension(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    real_mean = 0
    real_std = 1
    syn_mean = np.array((6, 0))
    syn_std = real_std

    dim_2_scale_list = np.logspace(-3, 3, 20)

    results = {}
    for dim_2_scale in dim_2_scale_list:
        real = stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, 2))
        real[:, 1] *= dim_2_scale
        metric_computer = metric_computer_factory(real)

        syn = stats.norm.rvs(loc=syn_mean, scale=syn_std, size=(n_real, 2))
        syn[:, 1] *= dim_2_scale

        metrics = metric_computer.compute_metric(syn)
        results[dim_2_scale] = metrics

    return results


def run_gaussian_mean_difference_with_pareto(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    real_mean = 0
    real_std = 1
    syn_means = np.linspace(-6, 6, 51)
    syn_std = real_std
    pareto_b = 1.01


    results = {}
    real1 = stats.norm.rvs(loc=real_mean, scale=real_std, size=n_real)
    real2 = stats.pareto.rvs(b=pareto_b, size=n_real)
    real = np.stack((real1, real2), axis=1)
    metric_computer = metric_computer_factory(real)
    for syn_mean in syn_means:
        syn1 = stats.norm.rvs(loc=syn_mean, scale=syn_std, size=n_syn)
        syn2 = stats.pareto.rvs(b=pareto_b, size=n_syn)
        syn = np.stack((syn1, syn2), axis=1)

        metrics = metric_computer.compute_metric(syn)
        results[syn_mean] = metrics

    return results


def run_high_dim_gaussian_one_disjoint_dim(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    real_mean = 0
    real_std = 1
    syn_mean_dim_1 = 6
    syn_mean_other_dims = real_mean
    syn_std = real_std

    extra_dim_list = np.logspace(0, 3, 20).astype(int)

    results = {}
    for extra_dim in extra_dim_list:
        real = stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, extra_dim + 1))
        metric_computer = metric_computer_factory(real)

        syn = stats.norm.rvs(loc=syn_mean_other_dims, scale=syn_std, size=(n_syn, extra_dim + 1))
        syn[:, 0] += syn_mean_dim_1 - syn_mean_other_dims

        metrics = metric_computer.compute_metric(syn)
        results[extra_dim] = metrics

    return results