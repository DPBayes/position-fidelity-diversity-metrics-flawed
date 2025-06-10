import numpy as np
import pandas as pd 
import os.path

from experiment_config import sanity_check_result_level_names, metric_types


def extract_sanity_check_name(table_filename):
    basename = os.path.basename(table_filename) 
    filename, _ = os.path.splitext(basename) 
    return filename


def bool_converter(v):
    if type(v) is bool:
        return v
    elif type(v) is str:
        if v == "True": return True
        elif v == "False": return False
        else: raise ValueError("Cannot convert {} to bool".format(v))
    else:
        raise ValueError("v must be str or bool, got {}".format(type(v)))


def pass_fail_aggregator(results):
    is_low_in_results = "low" in list(results)
    is_high_in_results = "high" in list(results)
    if is_low_in_results and is_high_in_results:
        return False
    results_without_low_high_nan = [
        bool_converter(res) for res in results if res != "high" and res != "low" and res is not np.nan
    ]
    are_all_boolean_true = all(results_without_low_high_nan)
    if is_low_in_results and are_all_boolean_true:
        return "low"
    elif is_high_in_results and are_all_boolean_true:
        return "high"
    else:
        return are_all_boolean_true


print_metric_names = {
    "precision": "I-Prec",
    "recall": "I-Rec",
    "density": "Density",
    "coverage": "Coverage",
    "integrated_alpha_precision": "IAP",
    "integrated_beta_recall": "IBR",
    "precision_cover": "C-Prec",
    "recall_cover": "C-Rec",
    "sym_precision": "symPrec",
    "sym_recall": "symRec",
    "probabilistic_precision": "P-Prec",
    "probabilistic_recall": "P-Rec",
}
fail_str = "\\color{red} F"
print_values = {
    "True": "T",
    True: "T",
    "False": fail_str,
    False: fail_str,
    "low": "L",
    "high": "H",
}
print_desiderata = {
    "property": "D1b (purpose)",
    "hyperparameters": "D2 (hyperparameters)",
    "real_data_size": "D3 (data)",
    "bounds": "D4 (bounds)",
    "transformations": "D5 (invariance)",
}
print_sanity_check_names = {
    "gaussian-mean-difference": "Gaussian Mean Difference",
    "gaussian-mean-difference-with-outlier" : "Gaussian Mean Difference + Outlier",
    "gaussian-std-difference": "Gaussian Std. Deviation Difference",
    "gaussian-mixture-simultaneous-mode-dropping": "Simultaneous Mode Dropping",
    "gaussian-mixture-sequential-mode-dropping": "Sequential Mode Dropping",
    "gaussian-mixture-mode-dropping-invention": "Mode Dropping + Invention",
    "uniform-hypersphere-surface": "Hypersphere Surface",
    "uniform-hypercube-varying-size": "Hypercube, Varying Sample Size",
    "uniform-hypercube-varying-syn-size": "Hypercube, Varying Syn. Size",
    "sphere-torus": "Sphere vs. Torus",
    "one-vs-two-modes": "Mode Collapse",
    "gaussian-scaling-one-dimension": "Scaling One Dimension",
    "gaussian-mean-difference-with-pareto": "Gaussian Mean Difference + Pareto",
    "gaussian-high-dim-one-disjoint-dim": "One Disjoint Dim. + Many Identical Dim.",
    "discrete-numerical-vs-continuous-numerical": "Discrete Num. vs. Continuous Num.",
}

tabular_sanity_check_names = [
    "discrete-numerical-vs-continuous-numerical",
    "gaussian-mean-difference-with-pareto"
]

def save_latex(df, filename):
    df.columns = df.columns.map(print_metric_names)
    df = df.replace(print_values)
    df = df.swaplevel()
    df.index.names = ["Desiderata", "Sanity Check"]
    df.insert(0, "Tab.", "")
    # populate the first column with "\checkmark" for sanity checks with names in list
    # and " " for those not in the list
    df.iloc[:, 0] = df.index.get_level_values(1).map(lambda x: "\\checkmark" if x in tabular_sanity_check_names else " ")
    df.rename(index=print_desiderata, level=0, inplace=True)
    df.rename(index=print_sanity_check_names, level=1, inplace=True)
    df = df.sort_index()
    df.style.to_latex(filename, hrules=True, clines="skip-last;data")

if __name__ == "__main__":

    fidelity_metrics = [metric for metric, metric_type in metric_types.items() if metric_type == "fidelity"]
    diversity_metrics = [metric for metric, metric_type in metric_types.items() if metric_type == "diversity"]

    sanity_check_name_list = []
    groupdf_list = []

    for pass_fail_table_filename in snakemake.input.pass_fail_tables:
        sanity_check_name = extract_sanity_check_name(pass_fail_table_filename)
        n_index_variables = len(sanity_check_result_level_names[sanity_check_name]) - 1 + 2

        pass_fail_df = pd.read_csv(str(pass_fail_table_filename), index_col=list(range(n_index_variables)))
        groupdf = pass_fail_df.groupby(level="desiderata").agg(pass_fail_aggregator)

        sanity_check_name_list.append(sanity_check_name)
        groupdf_list.append(groupdf)

    merged_df = pd.concat(groupdf_list, keys=sanity_check_name_list, names=["sanity check"])

    fidelity_df = merged_df.loc[:, [metric for metric in fidelity_metrics if metric in merged_df.columns]]
    diversity_df = merged_df.loc[:, [metric for metric in diversity_metrics if metric in merged_df.columns]]

    merged_df.to_csv(str(snakemake.output.metric_table))
    fidelity_df.to_csv(str(snakemake.output.fidelity_metric_table))
    diversity_df.to_csv(str(snakemake.output.diversity_metric_table))
    save_latex(fidelity_df, str(snakemake.output.fidelity_metric_table_latex))
    save_latex(diversity_df, str(snakemake.output.diversity_metric_table_latex))