import torch 
import numpy as np
from metrics import MetricComputer

from .representations.OneClass import OneClassLayer
from .evaluation import compute_alpha_precision

class AlphaPrecisionBetaRecallComputer(MetricComputer):
    def __init__(self, real_data):
        """Constructor for AlphaPrecisionBetaRecallComputer.

        Hyperparameters are set to default values from the original 
        implementation for their sanity checks.
        Args:
            real_data: Real data.
        """
        super().__init__(real_data)

        nn_params  = dict({"rep_dim": None, 
                        "num_layers": 2, 
                        "num_hidden": 200, 
                        "activation": "ReLU",
                        "dropout_prob": 0.5, 
                        "dropout_active": False,
                        "train_prop" : 1,
                        "epochs" : 100,
                        "warm_up_epochs" : 10,
                        "lr" : 1e-3,
                        "weight_decay" : 1e-2,
                        "LossFn": "SoftBoundary"})   
        hyperparams = dict({"Radius": 1, "nu": 1e-2})

        nn_params["input_dim"] = real_data.shape[1]
        nn_params["rep_dim"] = real_data.shape[1]        
        hyperparams["center"] = torch.ones(real_data.shape[1])

        self.model = OneClassLayer(params=nn_params, hyperparams=hyperparams)
        self.model.fit(self.real_data_scaled, verbosity=False)
        self.real_data_features = self.model(torch.tensor(self.real_data_scaled).float()).float().detach().numpy()

    def compute_metric(self, syn_data):
        syn_data_scaled = self.scale_data(syn_data)
        syn_data_features = self.model(torch.tensor(syn_data_scaled).float()).float().detach().numpy()

        alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authen = compute_alpha_precision(self.real_data_features, syn_data_features, self.model.c)
        return dict(
            integrated_alpha_precision=float(Delta_precision_alpha),
            integrated_beta_recall=float(Delta_coverage_beta),
        )
