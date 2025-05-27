import numpy as np 
from scipy import stats


def run_discrete_numerical_vs_continuous_numerical(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    real_mean = 0
    real_std = 1
    syn_mean = real_mean
    syn_std = real_std

    scale_list = np.logspace(0, 3, 20)
    
    results = {}
    for real_type in ["real_discrete", "real_continuous"]:
        results[real_type] = {}

        for scale in scale_list:
            real = scale * stats.norm.rvs(loc=real_mean, scale=real_std, size=(n_real, 1))
            if real_type == "real_discrete":
                real = np.round(real)
            metric_computer = metric_computer_factory(real)

            syn = scale * stats.norm.rvs(loc=syn_mean, scale=syn_std, size=(n_syn, 1))
            if real_type == "real_continuous":
                syn = np.round(syn)
            metrics = metric_computer.compute_metric(syn)
            results[real_type][scale] = metrics

    return results