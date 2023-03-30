"""
Language Translation Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking
dataset's Non-English splits
Original sources cited in the project's README

Example usage:
python cache_multilingual_fact_checking_dataset.py --hugging_face False
"""

import os
import json
import pandas as pd
import copy
from argparse import ArgumentParser
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize


def main(args):
    print("Translating the fact_checking dataset into Non-English languages...")

    pd_df_dict = {}
    dataset = load_dataset("CalibraGPT/Fact_Checking", split="English")
    print(dataset[0], "\n")

    for i in range(20):  # len(dataset)):
        true_pair = dataset[i]["stem"] + " " + dataset[i]["true"]
        counterfacts_list = (
            dataset[i]["false"]
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .split(", ")
        )

        false_pair_list = []
        for counterfact in counterfacts_list:
            false_pair_list.append(dataset[i]["stem"] + " " + counterfact)

        translated_true = GoogleTranslator(source="en", target="fr").translate(
            true_pair
        )
        translated_false_list = []
        for false_pair in false_pair_list:
            translated_false_list.append(
                GoogleTranslator(source="en", target="fr").translate(false_pair_list[0])
            )

        print(f"{true_pair} -> {translated_true}")
        translated_true_tokenized = word_tokenize(translated_true, language="french")
        print(translated_true_tokenized)

        translated_false_tokenized_list = []
        for itr, false_pair in enumerate(false_pair_list):
            print(f"{false_pair} -> {translated_false_list[itr]}")
            translated_false_tokenized_list.append(
                word_tokenize(translated_false_list[itr], language="french")
            )
            print(translated_false_tokenized_list[itr])

        same_stem = True
        for itr, false_pair_tokenized in enumerate(translated_false_tokenized_list):
            if false_pair_tokenized[:-1] != translated_true_tokenized[:-1]:
                same_stem = False
        if same_stem:
            print(" ".join(translated_true_tokenized[:-1]))
        else:
            stems = []
            stems.append(" ".join(translated_true_tokenized[:-1]))
            for false_pair_tokenized in translated_false_tokenized_list:
                stems.extend(" ".join(false_pair_tokenized[:-1]))

        # add logic to ignore if any of the objects don't end at the end here

        # if logic passes:

        try:
            pd_df_dict["dataset_id"].append(dataset[i]["dataset_id"])
        except KeyError:
            pd_df_dict["dataset_id"] = [dataset[i]["dataset_id"]]
        try:
            pd_df_dict["relation"].append(dataset[i]["relation"])
        except KeyError:
            pd_df_dict["relation"] = [dataset[i]["relation"]]

        print("\n")
    # 'stem': 'The location of Rosstown Railway is', 'true': 'Melbourne', 'false': "['England']", 'subject': 'Rosstown Railway', 'object': 'Melbourne'}

    print(pd_df_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hugging_face",
        type=bool,
        default=False,
        help="Whether or not to write to Hugging Face (access required)",
    )

    args = parser.parse_args()
    main(args)
