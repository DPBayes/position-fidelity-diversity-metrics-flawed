import pandas as pd
import numpy as np

from experiment_config import sanity_check_result_level_names
from sanity_check_criteria import *



sanity_check_criteria = {
    "gaussian-mean-difference": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("syn_mean", 0.0, check_close_to_one),
                "extreme negative": SimpleCriterion("syn_mean", "min", check_close_to_zero),
                "extreme positive": SimpleCriterion("syn_mean", "max", check_close_to_zero),
            },
            "property": {
                "bell shape": BellShapeCriterion("syn_mean", 0.0),
            },
        }
    ),
    "gaussian-mean-difference-with-outlier": SanityCheckCriteria(
        index_variables=["outlier_scenario", "dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("syn_mean", 0.0, check_close_to_one),
                "extreme negative": SimpleCriterion("syn_mean", "min", check_close_to_zero),
                "extreme positive": SimpleCriterion("syn_mean", "max", check_close_to_zero),
            },
            "property": {
                "bell shape": BellShapeCriterion("syn_mean", 0.0),
            },
        }
    ),
    "gaussian-std-difference": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("syn_std", 1.0, check_close_to_one),
                "low syn. std.": CriterionSeparatorByMetricType({
                    "fidelity": SimpleCriterion("syn_std", "min", check_close_to_one),
                    "diversity": SimpleCriterion("syn_std", "min", check_close_to_zero),
                }),
                "high syn. std.": CriterionSeparatorByMetricType({
                    "fidelity": SimpleCriterion("syn_std", "max", check_close_to_zero),
                    "diversity": SimpleCriterion("syn_std", "max", check_wide_syn_distribution_diversity)
                })
            },
            "property": {
                "shape": CriterionSeparatorByMetricType({
                    "fidelity": HighToLowShapeCriterion("syn_std"),
                    "diversity": DiversityHighLowShapeCriterion("syn_std", low_shape_bell_midpoint=1.0),
                })
            },
        }
    ),
    "gaussian-mixture-simultaneous-mode-dropping": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("drop_fraction", "min", check_close_to_one),
                "fully_dropped_modes": CriterionSeparatorByMetricType({
                    "fidelity": SimpleCriterion("drop_fraction", "max", check_close_to_one),
                    # Only checking fidelity metrics, not obvious what value should be for diversity metrics
                })
            },
            "property": {
                "shape": CriterionSeparatorByMetricType({
                    "fidelity": HorizontalLineShapeCriterion("drop_fraction"),
                    "diversity": HighToLowShapeWithMidDropCriterion(
                        "drop_fraction", midpoint_quantile=0.95, 
                        left_extreme_to_midpoint_min_difference=0.10), 
                })
            },
        }
    ),
    "gaussian-mixture-sequential-mode-dropping": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("n_dropped_modes", "min", check_close_to_one),
                "fully_dropped_modes": CriterionSeparatorByMetricType({
                    "fidelity": SimpleCriterion("n_dropped_modes", "max", check_close_to_one),
                    # Only checking fidelity metrics, not obvious what value should be for diversity metrics
                })
            },
            "property": {
                "shape": CriterionSeparatorByMetricType({
                    "fidelity": HorizontalLineShapeCriterion("n_dropped_modes"),
                    "diversity": HighToLowShapeWithMidDropCriterion(
                        "n_dropped_modes", midpoint_quantile=0.5, 
                        left_extreme_to_midpoint_min_difference=0.10), 
                })
            },
        }
    ),
    "gaussian-mixture-mode-dropping-invention": SanityCheckCriteria(
        index_variables=[],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("n_components_syn", 5, check_close_to_one),
                "most dropped modes": CriterionSeparatorByMetricType({
                    "fidelity": SimpleCriterion("n_components_syn", "min", check_close_to_one)
                }),
                "most invented modes": CriterionSeparatorByMetricType({
                    "diversity": SimpleCriterion("n_components_syn", "max", check_close_to_one)
                }),
            },
            "property": {
                "shape": CriterionSeparatorByMetricType({
                    "fidelity": HighToLowShapeCriterion("n_components_syn"),
                    "diversity": LowToHighShapeCriterion("n_components_syn"),
                })
            },
        }
    ),
    "uniform-hypersphere-surface": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("syn_radius", 1.0, check_close_to_one),
                "small syn. radius": SimpleCriterion("syn_radius", "min", check_close_to_zero),
                "large syn. radius": SimpleCriterion("syn_radius", "max", check_close_to_zero),
            },
            "property": {
                "shape": BellShapeCriterion("syn_radius", 1.0)
            }
        }
    ),
    "uniform-hypercube-varying-size": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "real_data_size": {
                "shape": ConvergingLineShapeCriterion("n", 0.5)
            },
            "property": {
                "large size": SimpleCriterion("n", "max", check_close_to(0.2)),
            }
        }
    ),
    "uniform-hypercube-varying-syn-size": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "hyperparameters": {
                "shape": ConvergingLineShapeCriterion("n_syn", 0.5)
            },
            "property": {
                "large syn. size": SimpleCriterion("n_syn", "max", check_close_to(0.2)),
            }
        }
    ),
    "sphere-torus": SanityCheckCriteria(
        index_variables=["real_dist"],
        criteria={
            "property": {
                "shape": ConvergingLineShapeCriterion("n_syn", 0.5),
            },
            "bounds": {
                "right extreme value": SimpleCriterion("n_syn", "max", check_close_to_zero),
            }
        }
    ),
    "one-vs-two-modes": SanityCheckCriteria(
        index_variables=["dim"],
        criteria={
            "property": {
                "shape": CriterionSeparatorByMetricType({
                    "fidelity": HighToLowShapeCriterion("mu"),
                    "diversity": DiversityHighLowShapeCriterion(
                        "mu", low_shape=HighToLowShapeCriterion("mu"),
                        high_shape=HorizontalLineShapeCriterion("mu"),
                    )
                })
            },
            "bounds": {
                "identical": SimpleCriterion("mu", "min", check_close_to_one),
            }
        }
    ),
    "gaussian-scaling-one-dimension": SanityCheckCriteria(
        index_variables=[],
        criteria={
            "transformations": {
                "shape": HorizontalLineShapeCriterion("dim_2_scale"),
            },
            "bounds": {
                "left extreme value": SimpleCriterion("dim_2_scale", "min", check_close_to_zero),
                "right extreme value": SimpleCriterion("dim_2_scale", "max", check_close_to_zero),
            }
        }
    ),
    "gaussian-mean-difference-with-pareto": SanityCheckCriteria(
        index_variables=[],
        criteria={
            "bounds": {
                "identical": SimpleCriterion("syn_mean", 0.0, check_close_to_one),
                "extreme negative": SimpleCriterion("syn_mean", "min", check_close_to_zero),
                "extreme positive": SimpleCriterion("syn_mean", "max", check_close_to_zero),
            },
            "property": {
                "bell shape": BellShapeCriterion("syn_mean", 0.0),
            },
        }
    ),
    "gaussian-high-dim-one-disjoint-dim": SanityCheckCriteria(
        index_variables=[],
        criteria={
            "property": {
                "shape": HorizontalLineShapeCriterion("extra_dim"),
            },
            "bounds": {
                "left extreme value": SimpleCriterion("extra_dim", "min", check_close_to_zero),
                "right extreme value": SimpleCriterion("extra_dim", "max", check_close_to_zero),
            }
        }
    ),
    "discrete-numerical-vs-continuous-numerical": SanityCheckCriteria(
        index_variables=["real_type"],
        criteria={
            "property": {
                "shape": HorizontalLineShapeCriterion("scale"),
            },
            "bounds": {
                "coarse discretisation": CriterionSeparatorByIndexValue("real_type", {
                    "real_discrete": CriterionSeparatorByMetricType({
                        "fidelity": SimpleCriterion("scale", "min", check_close_to_zero),
                        "diversity": SimpleCriterion("scale", "min", check_wide_syn_distribution_diversity),
                    }),
                    "real_continuous": CriterionSeparatorByMetricType({
                        "fidelity": SimpleCriterion("scale", "min", check_close_to_one),
                        "diversity": SimpleCriterion("scale", "min", check_close_to_zero),
                    }),
                }),
                "fine discretisation": CriterionSeparatorByIndexValue("real_type", {
                    "real_discrete": CriterionSeparatorByMetricType({
                        "fidelity": SimpleCriterion("scale", "max", check_close_to_zero),
                        "diversity": SimpleCriterion("scale", "max", check_wide_syn_distribution_diversity),
                    }),
                    "real_continuous": CriterionSeparatorByMetricType({
                        "fidelity": SimpleCriterion("scale", "max", check_close_to_one),
                        "diversity": SimpleCriterion("scale", "max", check_close_to_zero),
                    }),
                }),
            }
        }
    ),
}



if __name__ == "__main__":
    result_df = pd.read_csv(str(snakemake.input.result_table), index_col=False)
    sanity_check_name = str(snakemake.wildcards.sanity_check)
    level_names = sanity_check_result_level_names[sanity_check_name]

    group_df = result_df.groupby(["Metric"] + level_names, as_index=False)["value"].agg(["mean", "std"])
    pass_fail_df = sanity_check_criteria[sanity_check_name].get_pass_fail_df(group_df)
    pass_fail_df.to_csv(str(snakemake.output))