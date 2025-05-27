import numpy as np
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class MetricComputer(ABC):
    """
    Base class for metric computers. Provides self.real_data_scaled that 
    contains the real data, and the scale_data method to scale the synthetic 
    data.

    Subclasses should cache any expensive computations on the real data in the
    constructor.
    """
    def __init__(self, real_data):
        self.real_data = real_data
        self.scaler = StandardScaler()
        self.real_data_scaled = self.scaler.fit_transform(real_data)

    def scale_data(self, data):
        return self.scaler.transform(data)

    @abstractmethod
    def compute_metric(self, syn_data):
        pass
