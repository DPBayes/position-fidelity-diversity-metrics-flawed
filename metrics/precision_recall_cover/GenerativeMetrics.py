'''
    Author: Fasil Cheema
    Purpose: Computes several generative metrics such as density and coverage, precision and recall, 
             improved prevision and improved recall, and our proposed metric cover precision and cover recall. 
'''

# Modified by Ossi Räisä (under GPL-3.0 license), 2024-11-25
# - remove non-precision recall cover metrics

import numpy as np
from scipy.stats import norm 
import sklearn
from sklearn.neighbors import NearestNeighbors


def PRCover(P,Q,k,C):
    # Computes the proposed cover precision and cover recall metrics

    # Obtains the number of samples in both samples sets P and Q
    num_P = P.shape[0]
    num_Q = Q.shape[0]

    # C factor is simply an integer where k' = Ck (originally set to 3)

    # Computes the NN of both P and Q
    nbrs_P = NearestNeighbors(n_neighbors=(C*k)+1, algorithm='kd_tree').fit(P)
    nbrs_Q = NearestNeighbors(n_neighbors=(C*k)+1, algorithm='kd_tree').fit(Q)

    # Returns KNN distances and indices for each data sample
    dist_P, ind_P = nbrs_P.kneighbors(P)
    dist_Q, ind_Q = nbrs_Q.kneighbors(Q)

    # Note that the knn returns the pt itself as 1NN so we discard first column
    dist_P = dist_P[:,1:]
    dist_Q = dist_Q[:,1:]
    ind_P  =  ind_P[:,1:]
    ind_Q  =  ind_Q[:,1:]

    # Intialize metric counter
    p_sum = 0
    r_sum = 0

    # Initialize array for values that have PR cover with just P just Q and joint support 
    # Starts as location of point, then radius of knn ball
    P_disjoint_Q_pts = 0 
    Q_disjoint_P_pts = 0 
    joint_supp_pts = 0 

    P_disjoint_Q_knn = 0  
    Q_disjoint_P_knn = 0 
    joint_supp_knn = 0 


    # Iterates through sample set P and checks if the number of set pts within the sample pt k-NN are above the desired number
    for i in range(num_P):
        return_val = PR_Cover_Indicator(P[i],Q,dist_P[i], C)
        if return_val == 1:
            p_sum += 1

            #Checks if the first point, so it properly sets up pts with joint support
            if type(joint_supp_pts) == int: 
                joint_supp_pts = P[i]
                joint_supp_knn = dist_P[i][len(dist_P[i])-1]
            else:
                joint_supp_pts = np.vstack([joint_supp_pts, P[i]])
                joint_supp_knn = np.vstack([joint_supp_knn, dist_P[i][len(dist_P[i])-1]])
        else:
            if type(P_disjoint_Q_pts) == int: 
                P_disjoint_Q_pts = P[i]
                P_disjoint_Q_knn = dist_P[i][len(dist_P[i])-1]
            else:
                P_disjoint_Q_pts = np.vstack([P_disjoint_Q_pts, P[i]])
                P_disjoint_Q_knn = np.vstack([P_disjoint_Q_knn, dist_P[i][len(dist_P[i])-1]])




    # Computes cover_precision (num times k-nn ball for pt is sufficiently mixed divided )
    cover_precision = p_sum/num_P

    # Iterates through sample set Q and checks if the number of set pts within the sample pt k-NN are above the desired number
    for j in range(num_Q): 
        return_val = PR_Cover_Indicator(Q[j],P,dist_Q[j], C)
        if return_val == 1:
            r_sum += 1
            
            if type(joint_supp_pts) == int: 
                joint_supp_pts = Q[j]
                joint_supp_knn = dist_Q[j][len(dist_Q[j])-1]
            else:
                joint_supp_pts = np.vstack([joint_supp_pts, Q[j]])
                joint_supp_knn = np.vstack([joint_supp_knn, dist_Q[j][len(dist_Q[j])-1]])
        else:
            if type(Q_disjoint_P_pts) == int: 
                Q_disjoint_P_pts = Q[j]
                Q_disjoint_P_knn = dist_Q[j][len(dist_Q[j])-1]
            else:
                Q_disjoint_P_pts = np.vstack([Q_disjoint_P_pts, Q[j]])
                Q_disjoint_P_knn = np.vstack([Q_disjoint_P_knn, dist_Q[j][len(dist_Q[j])-1]])

    # Computes cover_recall (num times k-nn ball for pt is sufficiently mixed divided )
    cover_recall = r_sum/num_Q

    return cover_precision, cover_recall, P_disjoint_Q_pts, P_disjoint_Q_knn, Q_disjoint_P_pts, Q_disjoint_P_knn, joint_supp_pts, joint_supp_knn

def PR_Cover_Indicator(sample_pt, sample_set, k_nn_set, C):
    # Indicator function that checks if the number of pts from the set that lie within the k-NN ball of
    # the input point exceeds the required number of neighbors ( which is based off of the C factor k' = Ck)

    # Obtain important info such as choice of k, num_nbrs which is the min num of pts within a k-nn ball
    k = len(k_nn_set)
    num_nbrs = k/C
    num_pts  = sample_set.shape[0]

    # Initialize counter for num pts within k-nn ball 
    set_pts_in_knn = 0 
    
    # Iterate through each pt in set and check if it lies within main pt's k-nn ball if so add to count
    for i in range(num_pts):
        curr_dist = np.linalg.norm(sample_set[i] - sample_pt)
        
        if curr_dist <= k_nn_set[k-1]:
            set_pts_in_knn += 1
    
    # Checks if the number of pts that are within k-nn ball of main pt is above threshold (num_nbrs) if so return 1
    if set_pts_in_knn >= num_nbrs:
        indicator_val = 1
    else:
        indicator_val = 0

    return indicator_val
