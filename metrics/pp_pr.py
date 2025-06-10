
# Source: https://github.com/kdst-team/Probablistic_precision_recall/blob/master/metric/pp_pr.py
# Modified by Ossi Räisä under MIT license, 2024-11-25
# - add ProbabilisticPrecisionRecallComputer class

import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from metrics import MetricComputer


class ProbabilisticPrecisionRecallComputer(MetricComputer):
    def __init__(self, real_data, a=1.2, kth=4):
        super().__init__(real_data)
        self.jobs = 8
        self.gpu = False
        self.a = a
        self.kth = kth

        real = self.real_data_scaled
        self.pairwise_distance_real = pairwise_distances(real, real, n_jobs = self.jobs, metric = 'l2')
        self.k_nearest_x = get_average_of_knn_distance(self.pairwise_distance_real, self.kth, gpu = self.gpu)

    def compute_metric(self, syn_data):
        syn_data_scaled = self.scale_data(syn_data)

        real = self.real_data_scaled
        fake = syn_data_scaled

        pairwise_distance_real = self.pairwise_distance_real
        pairwise_distance_fake = pairwise_distances(fake, fake, n_jobs = self.jobs, metric = 'l2')
        distance = pairwise_distances(real, fake, n_jobs = self.jobs, metric = 'l2')
        distance_t = np.transpose(distance)


        # Calculate maximum distance for f
        k_nearest_x = self.k_nearest_x
        k_nearest_y = get_average_of_knn_distance(pairwise_distance_fake, self.kth, gpu = self.gpu)
        
        # Derive PSR scoring rule
        PSR_X = get_scoring_rule_psr(distance, k_nearest_x, self.a, self.gpu)
        PSR_Y = get_scoring_rule_psr(distance_t, k_nearest_y, self.a, self.gpu)
        
        # Calculate P-precision and P-recall
        if self.gpu:
            p_precision = torch.mean(1.0 - PSR_X).cpu()
            p_recall = torch.mean(1.0 - PSR_Y).cpu()
        else:
            p_precision = np.mean(1.0 - PSR_X)
            p_recall = np.mean(1.0 - PSR_Y)
        
        return dict(
            probabilistic_precision=float(p_precision), probabilistic_recall=float(p_recall),
        )


def get_pairwise_distances(real, fake, jobs = 8, gpu = False):
    if not gpu:
        pairwise_distance_between_real = pairwise_distances(real, real, n_jobs = jobs, metric = 'l2')
        pairwise_distance_between_fake = pairwise_distances(fake, fake, n_jobs = jobs, metric = 'l2')
        pairwise_distance_between_real_fake = pairwise_distances(real, fake, n_jobs = jobs, metric = 'l2')
        pairwise_distance_between_fake_real = np.transpose(pairwise_distance_between_real_fake)
    else:
        pairwise_distance_between_real = torch.cdist(real, real, p = 2)
        pairwise_distance_between_fake = torch.cdist(fake, fake, p = 2)
        pairwise_distance_between_real_fake = torch.cdist(real, fake, p = 2)
        pairwise_distance_between_fake_real = torch.transpose(pairwise_distance_between_real_fake, 0, 1)
    
    return (pairwise_distance_between_real,
           pairwise_distance_between_fake,
           pairwise_distance_between_real_fake,
           pairwise_distance_between_fake_real)

def get_average_of_knn_distance(x, kth, gpu = False):
    if not gpu:
        indices = np.argpartition(x, kth + 1, axis = -1)[..., :kth+1]
        k_smallests = np.take_along_axis(x, indices, axis = -1)
        kth_values = k_smallests.max(axis = -1)
        k_nearest = np.mean(kth_values)
    else:
        k_nearest = torch.mean(torch.sort(x, dim = 1)[0][:, kth])

    return k_nearest

def get_scoring_rule_psr(distance, k_nearest, a, gpu = False):
    out_of_knearest = distance >= a * k_nearest
    f = 1 - distance / (a * k_nearest)
    f[out_of_knearest] = 0.0
    if not gpu:
        psr = np.prod(1.0 - f, axis = 0)
    else:
        psr = torch.prod(1.0 - f, dim = 0)
    return psr

def compute_pprecision_precall(real, fake, a = 1.2, kth = 4, gpu = False):
    """
    Main Calculation of P-precision and P-recall.
    Args:
        real : Embeddings of real sample (N X D)
        fake : Embeddings of fake sample (N X D)
        a : Hyperparameter which controls the size of hypersphere
        kth : Hyperparameter for KNN
    Retures:
        p-precision, p-recall 
    """
    if gpu:
        print('Calculate with GPU')
        real, fake = torch.Tensor(real).cuda(), torch.Tensor(fake).cuda()

    pairwise_distance_real, pairwise_distance_fake, distance, distance_t = get_pairwise_distances(real, fake, gpu = gpu)

    # Calculate maximum distance for f
    k_nearest_x = get_average_of_knn_distance(pairwise_distance_real, kth, gpu = gpu)
    k_nearest_y = get_average_of_knn_distance(pairwise_distance_fake, kth, gpu = gpu)
    
    # Derive PSR scoring rule
    PSR_X = get_scoring_rule_psr(distance, k_nearest_x, a, gpu)
    PSR_Y = get_scoring_rule_psr(distance_t, k_nearest_y, a, gpu)
    
    # Calculate P-precision and P-recall
    if gpu:
        p_precision = torch.mean(1.0 - PSR_X).cpu()
        p_recall = torch.mean(1.0 - PSR_Y).cpu()
    else:
        p_precision = np.mean(1.0 - PSR_X)
        p_recall = np.mean(1.0 - PSR_Y)
    
    return p_precision, p_recall
