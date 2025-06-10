n_repeats = 10

sanity_checks = [
    "gaussian-mean-difference",
    "gaussian-mean-difference-with-outlier",
    "gaussian-std-difference",
    "gaussian-mixture-simultaneous-mode-dropping",
    "gaussian-mixture-sequential-mode-dropping",
    "gaussian-mixture-mode-dropping-invention",
    "uniform-hypersphere-surface",
    "uniform-hypercube-varying-size",
    "uniform-hypercube-varying-syn-size",
    "sphere-torus",
    "one-vs-two-modes",
    "gaussian-scaling-one-dimension",
    "gaussian-mean-difference-with-pareto",
    "gaussian-high-dim-one-disjoint-dim",
    "discrete-numerical-vs-continuous-numerical",
]

metrics = [
    "prdc",
    "alpha-precision;beta-recall;authenticity",
    "precision-recall-cover",
    "sym-precision-recall",
    "probabilistic-precision-recall",
]

rule all:
    input: 
        result_tables=expand(
            "results/result-tables/{sanity_check}.csv",
            sanity_check=sanity_checks
        ),
        fidelity_metric_table="results/fidelity-pass-fails.csv",
        diversity_metric_table="results/diversity-pass-fails.csv",
        fidelity_metric_table_latex="figures/fidelity-result-table.tex",
        diversity_metric_table_latex="figures/diversity-result-table.tex",

rule run_sanity_check:
    output: "results/sanity-checks/{sanity_check}/{metric}_{repeat_ind}.p"
    script: "run_sanity_check.py"

rule compile_result_table:
    input: 
        results=[
            "results/sanity-checks/{{sanity_check}}/{metric}_{repeat_ind}.p".format(
                metric=metric, repeat_ind=repeat_ind
            )
            for metric in metrics for repeat_ind in range(n_repeats)
        ],
    output: "results/result-tables/{sanity_check}.csv"
    script: "compile_result_table.py"

rule check_pass_fails:
    input:
        result_table="results/result-tables/{sanity_check}.csv",
    output: "results/pass-fail-tables/{sanity_check}.csv"
    script: "check_pass_fails.py"

rule aggregate_pass_fails:
    input:
        pass_fail_tables=expand(
            "results/pass-fail-tables/{sanity_check}.csv",
            sanity_check=sanity_checks
        )
    output:
        metric_table="results/pass-fails.csv",
        fidelity_metric_table="results/fidelity-pass-fails.csv",
        diversity_metric_table="results/diversity-pass-fails.csv",
        fidelity_metric_table_latex="figures/fidelity-result-table.tex",
        diversity_metric_table_latex="figures/diversity-result-table.tex",
    script: "aggregate_pass_fail_tables.py"