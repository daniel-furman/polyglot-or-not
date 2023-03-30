"""
Language Translation Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking
dataset's Non-English splits
Original sources cited in the project's README

Example usage:
python cache_multilingual_fact_checking_dataset.py --hugging_face False
"""

import os
import re
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
    # print(dataset[0], "\n")

    for i in range(3):  # len(dataset)):
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
                GoogleTranslator(source="en", target="fr").translate(false_pair)
            )

        translated_true_tokenized = word_tokenize(translated_true, language="french")

        translated_false_tokenized_list = []
        for itr, false_pair in enumerate(false_pair_list):
            translated_false_tokenized_list.append(
                word_tokenize(translated_false_list[itr], language="french")
            )

        # grab values for all the different colunmns in fact_checking

        # grab the stems
        stems = []
        # start with the true fact
        true_fact_translated = GoogleTranslator(source="en", target="fr").translate(
            dataset[i]["true"]
        )
        # see if the french translated object is at the end of the sentence
        index_after = len(translated_true_tokenized) - len(word_tokenize(true_fact_translated, language="french"))
        if true_fact_translated.lower() in " ".join(translated_true_tokenized[index_after:]).lower():            
            pattern = re.compile(true_fact_translated, re.IGNORECASE)
            pattern = pattern.sub("", translated_true)
            if pattern[-1] == ' ':
                pattern = pattern[0:-1]
            print(true_fact_translated)
            print(translated_true)
            stems.append(pattern)
        # otherwise, check if the original english object is at the end of the sentence
        else:
            index_after = len(translated_true_tokenized) - len(word_tokenize(dataset[i]["true"], language="english"))
            if dataset[i]["true"].lower() in " ".join(translated_true_tokenized[index_after:]).lower():
                pattern = re.compile(dataset[i]["true"], re.IGNORECASE)
                pattern = pattern.sub("", translated_true)
                if pattern[-1] == ' ':
                    pattern = pattern[:-1]
                print(dataset[i]["true"])
                print(translated_true)
                stems.append(pattern)

        # if neither, delete this row
        print(stems)


        # now add stems for all the counterfacts
        # STOPPED DEV HERE
        for itr, counterfact in enumerate(counterfacts_list):
            #print(word_tokenize(counterfact, language="english"))
            false_translated = GoogleTranslator(source="en", target="fr").translate(
                counterfact
            )
            #print(word_tokenize(false_translated, language="french"))
            #print(translated_false_tokenized_list[itr])

        same_stem = True
        for itr, false_pair_tokenized in enumerate(translated_false_tokenized_list):
            if false_pair_tokenized[:-1] != translated_true_tokenized[:-1]:
                same_stem = False
        if same_stem:
            # print(" ".join(translated_true_tokenized[:-1]))
            pass
        else:
            stems = []
            stems.append(" ".join(translated_true_tokenized[:-1]))
            for false_pair_tokenized in translated_false_tokenized_list:
                stems.extend([" ".join(false_pair_tokenized[:-1])])
            # print(stems)
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
