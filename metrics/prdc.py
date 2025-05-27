"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

# Modified by Ossi Räisä:
# - use <= instead of < when comparing distances to compute the metrics
#   - this fixes an issue where all metrics would be 0 for purely discrete data
# - add PRDCDComputer class to cache real data nearest neighbours datastructure

import numpy as np
import sklearn.metrics
from metrics import MetricComputer

__all__ = ['compute_prdc', 'PRDCComputer']


def expected_coverage(n_real, n_syn, k):
    """Computes expected coverage for identical real and synthetic distributions.
    
    See https://proceedings.mlr.press/v119/naeem20a.html, Lemma 2.

    Args:
        n_real (int): Size of the real data.
        n_syn (int): Suze of the synthetic data.
        k (int): k parameter for nearest neighbours.

    Returns:
        float: Expected coverage.
    """
    log_numer = 0
    log_denom = 0
    for i in range(k):
        log_numer += np.log(n_real - (i + 1))
        log_denom += np.log(n_real + n_syn - (i + 1))
    return 1 - np.exp(log_numer - log_denom)

class PRDCComputer(MetricComputer):
    def __init__(self, real_data, nearest_k=3, choose_k_dc=True, syn_data_size_for_k_choice=None, max_k=20, expected_coverage_threshold=0.95):
        """Constructor for PRDCComputer.

        Args:
            real_data: Real data.
            nearest_k (int, optional): k parameter for nearest neighbours. Defaults to 3.
            choose_k_dc (bool, optional): Whether to choose k automatically for density and coverage. Defaults to True.
            syn_data_size_for_k_choice (_type_, optional): Synthetic data size for automatic k selection. Defaults to None.
            max_k (int, optional): Maximum k for automatic choice. Defaults to 20.
            expected_coverage_threshold (float, optional): Threshold of expected coverage with identical distribution for automatic k choice. Defaults to 0.95.
        """
        super().__init__(real_data)
        self.nearest_k = nearest_k
        self.real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
            self.real_data_scaled, nearest_k
        )

        self.choose_k_dc = choose_k_dc
        if choose_k_dc:
            n_real = real_data.shape[0]
            if syn_data_size_for_k_choice is not None:
                n_syn = syn_data_size_for_k_choice
            else:
                n_syn = n_real
            for k in range(1, max_k + 1):
                if expected_coverage(n_real, n_syn, k) > expected_coverage_threshold:
                    self.nearest_k_dc = k
                    break
        
        self.real_nearest_neighbour_distances_dc = compute_nearest_neighbour_distances(
            self.real_data_scaled, self.nearest_k_dc
        )

    def compute_metric(self, syn_data):
        syn_data_scaled = self.scale_data(syn_data)
        fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
            syn_data_scaled, self.nearest_k
        )
        distance_real_fake = compute_pairwise_distance(
            self.real_data_scaled, syn_data_scaled
        )

        precision = (
            distance_real_fake <=
            np.expand_dims(self.real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()

        recall = (
                distance_real_fake <=
                np.expand_dims(fake_nearest_neighbour_distances, axis=0)
        ).any(axis=1).mean()

        density = (1. / float(self.nearest_k_dc)) * (
                distance_real_fake <=
                np.expand_dims(self.real_nearest_neighbour_distances_dc, axis=1)
        ).sum(axis=0).mean()

        coverage = (
                distance_real_fake.min(axis=1) <=
                self.real_nearest_neighbour_distances_dc
        ).mean()

        return dict(precision=precision, recall=recall,
                    density=density, coverage=coverage)


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <=
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <=
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <=
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <=
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)