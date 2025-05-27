import numpy as np 
from scipy import stats


def sample_gaussian_mixture(means, scales, probs, size):
    n_components = means.shape[0]
    dim = means.shape[1]
    selected_inds = np.random.choice(n_components, size=size, replace=True, p=probs)
    return stats.norm.rvs(size=(size, dim), loc=means[selected_inds, :], scale=scales[selected_inds, :])


def run_gaussian_mode_dropping_simultaneous(metric_computer_factory):
    n_real = 1000
    n_syn = n_real
    n_components = 10

    dim_list = [1, 8, 64]

    base_means = np.linspace(0, 10, n_components)
    base_scale_by_dim = {
        1: 1 / 6,
        8: 1 / 3,
        64: 1,
    }
    real_probs = np.ones(n_components) / n_components
    drop_fractions = np.linspace(0, 1, 50)

    results = {}
    for dim in dim_list:
        results[dim] = {}

        means = np.tile(base_means, (dim, 1)).transpose()
        scales = np.ones((n_components, dim)) * base_scale_by_dim[dim]
        real = sample_gaussian_mixture(means, scales, real_probs, n_real)

        metric_computer = metric_computer_factory(real)
        for drop_fraction in drop_fractions:
            syn_probs = real_probs.copy()
            syn_probs[1:] *= (1 - drop_fraction)
            syn_probs /= syn_probs.sum()
            syn = sample_gaussian_mixture(means, scales, syn_probs, n_syn)

            metrics = metric_computer.compute_metric(syn)
            results[dim][drop_fraction] = metrics

    return results


def run_gaussian_mode_dropping_sequential(metric_computer_factory):
    n_real = 1000
    n_syn = n_real
    n_components = 10

    dim_list = [1, 8, 64]

    base_means = np.linspace(0, 10, n_components)
    base_scale_by_dim = {
        1: 1 / 6,
        8: 1 / 3,
        64: 1,
    }
    real_probs = np.ones(n_components) / n_components

    results = {}
    for dim in dim_list:
        results[dim] = {}

        means = np.tile(base_means, (dim, 1)).transpose()
        scales = np.ones((n_components, dim)) * base_scale_by_dim[dim]
        real = sample_gaussian_mixture(means, scales, real_probs, n_real)

        metric_computer = metric_computer_factory(real)
        for n_dropped_modes in range(n_components):
            syn_probs = real_probs.copy()
            syn_probs[n_components - n_dropped_modes:] = 0
            syn_probs /= syn_probs.sum()
            syn = sample_gaussian_mixture(means, scales, syn_probs, n_syn)

            metrics = metric_computer.compute_metric(syn)
            results[dim][n_dropped_modes] = metrics

    return results


def run_gaussian_mode_dropping_invention(metric_computer_factory):
    means = np.array([
        [  5.46794328,  -4.08481523],
        [  4.87211875, -13.64864684],
        [-16.39432073,   9.09802249],
        [ -2.15447941,  12.46200019],
        [ 11.55784483,  -4.96049923],
        [-24.60558474,  14.18020818],
        [ 16.84215967,  -3.18710751],
        [ -5.47807206,  -6.77247129],
        [  9.46726546,   1.05922347],
        [ -5.40346201, -17.29490131]
    ])
    scale = 0.25

    n_real = 1000
    n_syn = n_real
    n_components_real = 5

    real_probs = np.ones(n_components_real) / n_components_real

    results = {}

    dim = 2
    scales = np.ones((means.shape[0], dim)) * scale
    real = sample_gaussian_mixture(means[:n_components_real, :], scales[:n_components_real, :], real_probs, n_real)

    metric_computer = metric_computer_factory(real)
    for n_components_syn in range(1, means.shape[0] + 1):
        # This works, but could be done the same way as with real probabilities
        syn_probs = np.zeros(n_components_syn)
        syn_probs[:n_components_syn] = 1
        syn_probs /= syn_probs.sum()
        syn = sample_gaussian_mixture(means[:n_components_syn, :], scales[:n_components_syn, :], syn_probs, n_syn)

        metrics = metric_computer.compute_metric(syn)
        results[n_components_syn] = metrics

    return results


def run_one_vs_two_modes(metric_computer_factory):
    n_real = 1000
    n_syn = n_real

    dim_list = [1, 8, 64]

    mu_list = np.linspace(0, 5, 20)

    results = {}
    for dim in dim_list:
        results[dim] = {}
        for mu in mu_list:
            means = np.tile(np.array([-0.5 * mu, 0.5 * mu]), (dim, 1)).transpose()
            real = sample_gaussian_mixture(means, np.ones((2, dim)), np.array((0.5, 0.5)), n_real)
            metric_computer = metric_computer_factory(real)

            syn = stats.norm.rvs(loc=0, scale=np.sqrt(1 + mu**2), size=(n_syn, dim))
            metrics = metric_computer.compute_metric(syn)
            results[dim][mu] = metrics

    return results 

