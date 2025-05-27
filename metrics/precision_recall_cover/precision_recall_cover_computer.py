
# Most code here is from GenerativeMetrics.py

import numpy as np 
from sklearn.neighbors import NearestNeighbors
from .GenerativeMetrics import PR_Cover_Indicator
from metrics import MetricComputer


class PrecisionRecallCoverComputer(MetricComputer):
    def __init__(self, real_data, C=3, k=None, C_0=6):
        """Constructor for PrecisionRecallCoverComputer.

        Hyperparameters k and k' are described in https://proceedings.mlr.press/v206/cheema23a.html.
        Arguments set them to k' = Ck, and k such that 
        k' = log(n) + C_0, where n is the number of samples in the dataset
        by default.

        Args:
            real_data: Real data.
            C (int, optional): Defaults to 3.
            k (_type_, optional): Defaults to None.
            C_0 (int, optional): Defaults to 6.
        """
        super().__init__(real_data)
        # C factor is simply an integer where k' = Ck (originally set to 3)
        self.C = C 
        if k is None:
            k = np.log(real_data.shape[0]) + C_0
            k = int(k + 1)
            k = int(k/C + 1)

        self.k = k
        self.n_neighbors = (C * k) + 1
        self.nbrs_P = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree').fit(self.real_data_scaled)
        self.dist_P, self.ind_P = self.nbrs_P.kneighbors(self.real_data_scaled)

        # Note that the knn returns the pt itself as 1NN so we discard first column
        self.dist_P = self.dist_P[:,1:]
        self.ind_P  =  self.ind_P[:,1:]

    def compute_metric(self, syn_data):
        syn_data_scaled = self.scale_data(syn_data)

        num_P = self.real_data_scaled.shape[0]
        num_Q = syn_data_scaled.shape[0]

        nbrs_Q = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree').fit(syn_data_scaled)

        # Returns KNN distances and indices for each data sample
        dist_Q, ind_Q = nbrs_Q.kneighbors(syn_data_scaled)
        dist_Q = dist_Q[:,1:]
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

        P = self.real_data_scaled
        Q = syn_data_scaled
        C = self.C
        dist_P = self.dist_P

        # Iterates through sample set P and checks if the number of set pts within the sample pt k-NN are above the desired number
        for i in range(num_P):
            return_val = PR_Cover_Indicator(P[i], Q, dist_P[i], C)
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

        return dict(
            precision_cover=cover_precision, recall_cover=cover_recall,
        )



def PRCover(P,Q,k,C):
    # Computes the proposed cover precision and cover recall metrics

    # Obtains the number of samples in both samples sets P and Q

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

