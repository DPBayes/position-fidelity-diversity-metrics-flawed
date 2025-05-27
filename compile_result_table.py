import pickle
import numpy as np
from scipy import stats
import pandas as pd

from experiment_config import sanity_check_result_level_names

metric_groups = {
    "precision": 0,
    "recall": 0,
    "density": 0,
    "coverage": 0,
    "integrated_alpha_precision": 1,
    "integrated_beta_recall": 1,
    "authenticity": 1,
    "precision_cover": 1,
    "recall_cover": 1,
    "sym_precision": 2,
    "sym_recall": 2,
    "probabilistic_precision": 2,
    "probabilistic_recall": 2,
}


def result_record(metric_name, sanity_check_name, repeat_ind, value, level_names, level_keys):
    result_record = {
        "Metric": metric_name,
        "metric_group": metric_groups[metric_name],
        "Sanity Check": sanity_check_name,
        "repeat_ind": repeat_ind,
        "value": value,
    }
    for i, level_name in enumerate(level_names):
        result_record[level_name] = level_keys[i]
    return result_record

if __name__ == "__main__":
    sanity_check_name = str(snakemake.wildcards.sanity_check)

    result_records = []

    for filename in snakemake.input.results:
        with open(str(filename), "rb") as file:
            result_obj = pickle.load(file)
        result_dict = result_obj["results"]
        repeat_ind = result_obj["repeat_ind"]

        level_names = sanity_check_result_level_names[sanity_check_name]
        if len(level_names) == 1:
            for level1_key, level1_results in result_dict.items():
                for metric_name, value in level1_results.items():
                    result_records.append(result_record(
                        metric_name, sanity_check_name, repeat_ind, value, 
                        level_names, [level1_key]
                    ))
        elif len(level_names) == 2:
            for level1_key, level1_results in result_dict.items():
                for level2_key, level2_results in level1_results.items():
                    for metric_name, value in level2_results.items():
                        result_records.append(result_record(
                            metric_name, sanity_check_name, repeat_ind, value, 
                            level_names, [level1_key, level2_key]
                        ))
        elif len(level_names) == 3:
            for level1_key, level1_results in result_dict.items():
                for level2_key, level2_results in level1_results.items():
                    for level3_key, level3_results in level2_results.items():
                        for metric_name, value in level3_results.items():
                            result_records.append(result_record(
                                metric_name, sanity_check_name, repeat_ind, value, 
                                level_names, [level1_key, level2_key, level3_key]
                            ))
        else:
            raise ValueError("Length of level_names is {}".format(len(level_names)))

    result_df = pd.DataFrame.from_records(result_records)
    result_df.to_csv(str(snakemake.output), index=False)