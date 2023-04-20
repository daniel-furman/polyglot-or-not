"""
Process fact-completion logs to generate bootstrap estimates and error analysis data

Example usage:
python post_process_logs.py \
    --path /Users/danielfurman/Desktop/tmp_logs
"""

from typing import List
from random import choices
import numpy as np
import json
import pandas as pd
from argparse import ArgumentParser
import glob
import os
import csv
import sys


def post_process(args):
    num_resamples = args.num_resamples
    input_path = args.path

    # check if input is a file or folder
    if os.path.isfile(input_path):
        input = [input_path]
        output_file = os.path.splitext(input_path)[0] + '-metrics.csv'

    elif os.path.isdir(input_path):
        input = glob.glob(os.path.join(input_path, "*.json"))
        output_file = os.path.join(input_path, 'metrics.csv')
    else:
        print("Invalid input path!")
        return

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # write header row
        writer.writerow(['model_name', 'language', 'num_total', 'num_correct', 'uncertainty'])

        for input_file in input:
            with open(input_file, "r") as f:
                data = json.load(f)

            model_name = data["model_name"][0]
            language = supported_languages[input_file.split('/')[-1].split('-')[0]]

            print(f"\n\tThe model name is {model_name}")
            print(
                f"\tThe language is {language}"
            )

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
                    false_facts_list.append(
                        [fact["stem"], [fact["fact"], fact["counterfact"]]]
                    )
                    false_facts["differences"].append(
                        fact["p_true"] - fact["p_false_average"]
                    )
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
            num_total = len(results)
            num_correct = np.round(100 * np.sum(results) / len(results), decimals=3)
            print(f"\tThere are {num_total} stem/fact pairs in the log")
            print(
                f"\tThe model got {num_correct}% of facts correct"
            )

            
            # create bootstrap estimates from logs
            # calculate percentage with this to check

            bootstrap_results = bootstrap(results, B=num_resamples)
            uncertainty = np.round(100 * bootstrap_results[0], decimals=3)
            writer.writerow([model_name, language, num_total, num_correct, uncertainty])

            print(
                f"\tThe 95% uncertainty estimate is +/- {uncertainty}%\n"
            )

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


supported_languages = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "de": "german",
    "uk": "ukrainian",
    "ro": "romanian",
    "bg": "bulgarian",
    "ca": "catalan",
    "da": "danish",
    "hr": "croatian",
    "hu": "hungarian",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovenian",
    "sr": "serbian",
    "sv": "swedish",
    "cs": "czech",
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Folder of logs or File",
    )
    parser.add_argument(
        "--num_resamples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples",
    )

    args = parser.parse_args()
    post_process(args)
