
from metrics import PRDCComputer, AlphaPrecisionBetaRecallComputer, PrecisionRecallCoverComputer, SymPrecisionRecallComputer
# from metrics import ProbabilisticPrecisionRecallComputer

from sanity_checks.gaussian import run_gaussian_mean_difference, run_gaussian_mean_difference_with_outlier
from sanity_checks.gaussian import run_gaussian_std_difference
from sanity_checks.gaussian_mixture import run_gaussian_mode_dropping_simultaneous
from sanity_checks.gaussian_mixture import run_gaussian_mode_dropping_sequential
from sanity_checks.gaussian_mixture import run_gaussian_mode_dropping_invention
from sanity_checks.hypersphere_surface import run_uniform_hypersphere_surface
from sanity_checks.hypercube import run_uniform_hypercube_varying_size
from sanity_checks.hypercube import run_uniform_hypercube_varying_syn_size
from sanity_checks.torus import run_sphere_torus
from sanity_checks.gaussian_mixture import run_one_vs_two_modes
from sanity_checks.gaussian import run_gaussian_scaling_one_dimension
from sanity_checks.gaussian import run_gaussian_mean_difference_with_pareto
from sanity_checks.gaussian import run_high_dim_gaussian_one_disjoint_dim
from sanity_checks.discrete import run_discrete_numerical_vs_continuous_numerical

metric_computers = {
    "prdc": PRDCComputer,
    "alpha-precision;beta-recall;authenticity": AlphaPrecisionBetaRecallComputer,
    "precision-recall-cover": PrecisionRecallCoverComputer,
    "sym-precision-recall": SymPrecisionRecallComputer,
    # "probabilistic-precision-recall": ProbabilisticPrecisionRecallComputer,
}

metrics = list(metric_computers.keys())

metric_types = {
    "precision": "fidelity",
    "recall": "diversity",
    "density": "fidelity",
    "coverage": "diversity",
    "integrated_alpha_precision": "fidelity",
    "integrated_beta_recall": "diversity",
    "precision_cover": "fidelity",
    "recall_cover": "diversity",
    "sym_precision": "fidelity",
    "sym_recall": "diversity",
    # "probabilistic_precision": "fidelity",
    # "probabilistic_recall": "diversity",
}

sanity_checks = {
    "gaussian-mean-difference": run_gaussian_mean_difference,
    "gaussian-mean-difference-with-outlier": run_gaussian_mean_difference_with_outlier,
    "gaussian-std-difference": run_gaussian_std_difference,
    "gaussian-mixture-sequential-mode-dropping": run_gaussian_mode_dropping_sequential,
    "gaussian-mixture-simultaneous-mode-dropping": run_gaussian_mode_dropping_simultaneous,
    "gaussian-mixture-mode-dropping-invention": run_gaussian_mode_dropping_invention,
    "uniform-hypersphere-surface": run_uniform_hypersphere_surface,
    "uniform-hypercube-varying-size": run_uniform_hypercube_varying_size,
    "uniform-hypercube-varying-syn-size": run_uniform_hypercube_varying_syn_size,
    "sphere-torus": run_sphere_torus,
    "one-vs-two-modes": run_one_vs_two_modes,
    "gaussian-scaling-one-dimension": run_gaussian_scaling_one_dimension,
    "gaussian-mean-difference-with-pareto": run_gaussian_mean_difference_with_pareto,
    "gaussian-high-dim-one-disjoint-dim": run_high_dim_gaussian_one_disjoint_dim,
    "discrete-numerical-vs-continuous-numerical": run_discrete_numerical_vs_continuous_numerical,
}

sanity_check_result_level_names = {
    "gaussian-mean-difference": ["dim", "syn_mean"],
    "gaussian-mean-difference-with-outlier": ["dim", "syn_mean", "outlier_scenario"],
    "gaussian-std-difference": ["dim", "syn_std"],
    "gaussian-mixture-simultaneous-mode-dropping": ["dim", "drop_fraction"],
    "gaussian-mixture-sequential-mode-dropping": ["dim", "n_dropped_modes"],
    "gaussian-mixture-mode-dropping-invention": ["n_components_syn"],
    "uniform-hypersphere-surface": ["dim", "syn_radius"],
    "uniform-hypercube-varying-size": ["dim", "n"],
    "uniform-hypercube-varying-syn-size": ["dim", "n_syn"],
    "sphere-torus": ["real_dist", "n_syn"],
    "one-vs-two-modes": ["dim", "mu"],
    "gaussian-scaling-one-dimension": ["dim_2_scale"],
    "gaussian-mean-difference-with-pareto": ["syn_mean"],
    "gaussian-high-dim-one-disjoint-dim": ["extra_dim"],
    "discrete-numerical-vs-continuous-numerical": ["real_type", "scale"],
}