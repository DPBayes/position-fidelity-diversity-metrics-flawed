# Evaluating Synthetic Data Evaluation Metrics

## Installing Dependencies
We use [Poetry](https://python-poetry.org/) to manage dependencies. Install Poetry 
according to their instructions, and use
```
poetry install
```
to install all our dependencies.

## Running Sanity Checks

We use [Snakemake](https://snakemake.github.io/) to run our code. The command
```
poetry run snakemake -j 5
```
runs all sanity checks and compiles their results (Tables 3 and 4 in the paper).
This uses 5 processes in parallel, controlled by the `-j 5` argument.

## Plotting 

Notebooks that plot all of our figures are in the `plotting` directory.

## External Code Sources and Licenses
The following code is taken from external sources, with minor modifications. Note that these are licensed under their original licenses, as specified below.


- Precision, recall, density, coverage in `metrics/prdc.py`: [https://github.com/clovaai/generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc), MIT license
- Alpha-precision, beta-recall in `metrics/alpha_precision_beta_recall`: [https://github.com/ahmedmalaa/evaluating-generative-models](https://github.com/ahmedmalaa/evaluating-generative-models), MIT license or BSD 3-clause license (repo has MIT, files say BSD 3-clause)
- precision and recall cover in `metrics/precision_recall_cover`: [https://github.com/FasilCheema/GenerativeMetrics](https://github.com/FasilCheema/GenerativeMetrics), GPL-3.0 license
- Symmetric precision and recall in `metrics/sym_precision_recall`: [https://github.com/mahyarkoy/emergent_asymmetry_pr](https://github.com/mahyarkoy/emergent_asymmetry_pr), MIT license

The paper also uses code from the following repository, which is not included at the moment due to the repository missing a license:
- Probabilistic precision and recall in `metrics/pp_pr.py` (not included at the moment): [https://github.com/kdst-team/Probablistic_precision_recall](https://github.com/kdst-team/Probablistic_precision_recall), no license given