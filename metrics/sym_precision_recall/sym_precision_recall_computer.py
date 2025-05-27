
import numpy as np 
from metrics import MetricComputer
from .manifmetric import ManifoldMetric


class SymPrecisionRecallComputer(MetricComputer):
    def __init__(self, real_data):
        super().__init__(real_data)
        self.manifmetric = ManifoldMetric(ref_data=self.real_data_scaled, k=5)

    def compute_metric(self, syn_data):
        syn_data_scaled = self.scale_data(syn_data)
        gen_stats = self.manifmetric.compute_stats(syn_data_scaled, k=5)
        sym_precision = self.manifmetric.sym_precision(gen_stats=gen_stats).sym_precision
        sym_recall = self.manifmetric.sym_recall(gen_stats=gen_stats).sym_recall
        return dict(
            sym_precision=sym_precision, sym_recall=sym_recall,
        )