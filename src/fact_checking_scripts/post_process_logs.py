"""
Process fact-checking logs to generate bootstrap estimates and error analysis data 

Example usage:
python post_process_logs.py --filename roberta-large-logged-cka-outputs-25-03-2023-17-28-56.json --num_resamples 100 --save_results True
"""

from typing import List
from random import choices
import numpy as np
import json
import pandas as pd
from argparse import ArgumentParser


def post_process(args):

    filename = args.filename
    num_resamples = args.num_resamples
    save_results = args.save_results

    # change input filename to the path to the log to be processed
    input_filename = f"../../src/fact_checking_scripts/output_logs/{filename}"
    print(f"Running post-processing for {input_filename}...")

    with open(input_filename, "r") as f:
        data = json.load(f)

    model_name = data["model_name"][0]
    print(f"\tThe model name is {model_name}")

    false_facts_itrs = []
    false_facts_list = []
    true_facts_itrs = []

    false_facts = {}
    false_facts["model"] = []
    false_facts["dataset_id"] = []
    false_facts["differences"] = []
    false_facts["facts"] = []
    false_facts["resolution"] = []

    for itr, fact in enumerate(data["score_dict_full"][model_name.lower()]):

        if fact["p_true > p_false_average"] != "True":
            false_facts_itrs.append(itr)
            false_facts_list.append([fact["stem"], [fact["fact"], fact["counterfact"]]])
            false_facts["differences"].append(fact["p_true"] - fact["p_false_average"])
            false_facts["facts"].append(
                f"{fact['stem']} [ true: {fact['fact']}; false: {fact['counterfact']} ]"
            )
            false_facts["model"].append(model_name)
            false_facts["dataset_id"].append(fact["dataset_id"])
            false_facts["resolution"].append("to do")

        elif fact["p_true > p_false_average"] == "True":
            true_facts_itrs.append(itr)
    # make a results list compatible with the bootstrap:

    results_false = [0] * len(false_facts_itrs)
    results_true = [1] * len(true_facts_itrs)
    results = results_false + results_true
    # should be ~33k
    print(f"\tThere are {len(results)} stem/fact pairs in the log")
    print(
        f"\tThe model got {np.round(100 * np.sum(results) / len(results), decimals=3)}% of facts correct"
    )

    # create bootstrap estimates from logs
    # calculate percentage with this to check

    bootstrap_results = bootstrap(results, B=num_resamples)

    print(
        f"\tThe 95% uncertainty estimate is +/- {np.round(100 * bootstrap_results[0], decimals=3)}%\n"
    )

    ## Grab items with the most negative p_false_average - p_true value
    # order results by p_true - p_false, return top n rows

    n = 100
    error_df = pd.DataFrame.from_dict(false_facts)
    error_df = error_df.sort_values(by="differences").head(n).reset_index(drop=True)
    error_df = error_df[["model", "dataset_id", "differences", "resolution", "facts"]]
    print(error_df)

    # optionally save:
    if save_results:
        error_df.to_csv(f"error-analysis-{filename}.csv", index=False)


def bootstrap(results: List[int], B: int = 10000, confidence_level: int = 0.95) -> int:

    """
    helper function for providing confidence intervals for sentiment tool
    """

    # compute lower and upper significance index
    critical_value = (1 - confidence_level) / 2
    lower_sig = 100 * critical_value
    upper_sig = 100 * (1 - critical_value)
    data = []
    for p in results:
        data.append(p)

    sums = []
    # bootstrap resampling loop
    for b in range(B):
        choice = choices(data, k=len(data))
        choice = np.array(choice)
        inner_sum = np.sum(choice) / len(choice)
        sums.append(inner_sum)

    percentiles = np.percentile(sums, [lower_sig, 50, upper_sig])

    lower = percentiles[0]
    median = percentiles[1]
    upper = percentiles[2]

    e_bar = ((median - lower) + (upper - median)) / 2
    return e_bar, median, percentiles


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        default="bert-tiny-logged-cka-outputs-26-03-2023-02-33-08.json",
        help="Filename of a log in 'output_logs/",
    )
    parser.add_argument(
        "--num_resamples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether or not to save error analysis results",
    )

    args = parser.parse_args()
    post_process(args)
