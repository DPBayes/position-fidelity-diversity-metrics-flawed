
import pickle
import hashlib
import numpy as np
import torch
import pandas as pd
from experiment_config import metric_computers, sanity_checks, sanity_check_result_level_names


def count_result_levels(results):
    """Count the number of nested dicts in input.

    Args:
        results (dict): Input dict.

    Returns:
        int: Number of nested dicts in input.
    """
    counter = 0
    obj = results
    while type(obj) is dict:
        obj = list(obj.values())[0]
        counter += 1
    return counter



def get_random_seed(*values):
    """Generate a random seed based on the input values.

    Returns:
        int: seed
    """

    list_of_hashes = [hashlib.md5(str(value).encode("utf-8"), usedforsecurity=False).digest() for value in values]
    ss = np.random.SeedSequence([int(hash.hex(), 16) for hash in list_of_hashes])
    return ss.generate_state(1)[0]


if __name__ == "__main__":
    metric_name = str(snakemake.wildcards.metric)
    sanity_check_name = str(snakemake.wildcards.sanity_check)
    repeat_ind = int(snakemake.wildcards.repeat_ind)

    seed = get_random_seed("run sanity check", metric_name, sanity_check_name, repeat_ind)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if metric_name not in metric_computers:
        raise Exception("Unknown metric {}".format(metric_name))
    metric_computer = metric_computers[metric_name]

    if sanity_check_name not in sanity_checks:
        raise Exception("Unknown sanity check {}".format(sanity_check_name))
    run_sanity_check = sanity_checks[sanity_check_name]
    results = run_sanity_check(metric_computer)

    if results is None:
        raise Exception("Sanity check {} did not return results.".format(sanity_check_name))
    if type(results) is not dict:
        raise Exception("Sanity check {} did not return a dict.".format(sanity_check_name))

    
    level_names = sanity_check_result_level_names[sanity_check_name]
    n_levels_in_results = count_result_levels(results)

    if len(level_names) != n_levels_in_results - 1:
        raise Exception(
            "Results for sanity check {} have the wrong number of levels. Expected {}, got {}".format(
                sanity_check_name, len(level_names), n_levels_in_results - 1
            )
        )


    output_obj = {
        "metric_name": metric_name,
        "sanity_check_name": sanity_check_name,
        "repeat_ind": repeat_ind,
        "results": results,
    }

    with open(str(snakemake.output), "wb") as file:
        pickle.dump(output_obj, file)