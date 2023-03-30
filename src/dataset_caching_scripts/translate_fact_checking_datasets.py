"""
Language Translation Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking
dataset's Non-English splits
Original sources cited in the project's README

Example usage:
python cache_multilingual_fact_checking_dataset.py --hugging_face False
"""

import re
import pandas as pd
import os
import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize


def main(args):
    print("Translating the fact_checking dataset into Non-English languages...")

    # nltk comp languages:
    languages = [("fr", "french")]

    for language in languages:

        pd_df_dict = {}
        dataset = load_dataset("CalibraGPT/Fact_Checking", split="English")

        for i in tqdm.tqdm(range(50)):  # tqdm.tqdm(range(len(dataset))):
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

            translated_true = GoogleTranslator(
                source="en", target=language[0]
            ).translate(true_pair)
            translated_false_list = []
            for false_pair in false_pair_list:
                translated_false_list.append(
                    GoogleTranslator(source="en", target=language[0]).translate(
                        false_pair
                    )
                )

            translated_true_tokenized = word_tokenize(
                translated_true, language=language[1]
            )

            translated_false_tokenized_list = []
            for itr, false_pair in enumerate(false_pair_list):
                translated_false_tokenized_list.append(
                    word_tokenize(translated_false_list[itr], language=language[1])
                )

            # grab values for all the different colunmns in fact_checking
            # grab the stems
            stems = []
            # start with the true fact
            true_fact_translated = GoogleTranslator(
                source="en", target=language[0]
            ).translate(dataset[i]["true"])
            # see if the translated object is at the end of the sentence
            index_fact = len(translated_true_tokenized) - len(
                word_tokenize(true_fact_translated, language=language[1])
            )
            if (
                true_fact_translated.lower()
                in " ".join(translated_true_tokenized[index_fact:]).lower()
            ):
                stem_pattern = re.compile(true_fact_translated, re.IGNORECASE)
                stem_pattern = stem_pattern.sub("", translated_true)
                if stem_pattern[-1] == " ":
                    stem_pattern = stem_pattern[0:-1]

                try:
                    pd_df_dict["true"].append(true_fact_translated)
                except KeyError:
                    pd_df_dict["true"] = [true_fact_translated]

                stems.append(stem_pattern)
            # otherwise, check if the original english object is at the end of the sentence
            else:
                index_fact = len(translated_true_tokenized) - len(
                    word_tokenize(dataset[i]["true"], language="english")
                )
                if (
                    dataset[i]["true"].lower()
                    in " ".join(translated_true_tokenized[index_fact:]).lower()
                ):
                    stem_pattern = re.compile(dataset[i]["true"], re.IGNORECASE)
                    stem_pattern = stem_pattern.sub("", translated_true)
                    if stem_pattern[-1] == " ":
                        stem_pattern = stem_pattern[:-1]
                    stems.append(stem_pattern)
                try:
                    pd_df_dict["true"].append(dataset[i]["true"])
                except KeyError:
                    pd_df_dict["true"] = [dataset[i]["true"]]

            counterfact_save_list = []
            # now add data for all the counterfacts
            for itr, counterfact in enumerate(counterfacts_list):
                false_fact_translated = GoogleTranslator(
                    source="en", target=language[0]
                ).translate(counterfact)
                # print(translated_false_list[itr])
                # see if the french translated object is at the end of the sentence
                index_fact = len(translated_false_tokenized_list[itr]) - len(
                    word_tokenize(false_fact_translated, language=language[1])
                )
                if (
                    false_fact_translated.lower()
                    in " ".join(
                        translated_false_tokenized_list[itr][index_fact:]
                    ).lower()
                ):
                    counterfact_save_list.append(false_fact_translated)
                    stem_pattern = re.compile(false_fact_translated, re.IGNORECASE)
                    stem_pattern = stem_pattern.sub("", translated_false_list[itr])
                    if stem_pattern[-1] == " ":
                        stem_pattern = stem_pattern[0:-1]

                    stems.append(stem_pattern)

                # otherwise, check if the original english object is at the end of the sentence
                else:

                    index_fact = len(translated_false_tokenized_list[itr]) - len(
                        word_tokenize(counterfacts_list[itr], language="english")
                    )

                    if (
                        counterfacts_list[itr]
                        in " ".join(
                            translated_false_tokenized_list[itr][index_fact:]
                        ).lower()
                    ):
                        counterfact_save_list.append(counterfacts_list[itr])
                        stem_pattern = re.compile(counterfacts_list[itr], re.IGNORECASE)
                        stem_pattern = stem_pattern.sub("", translated_false_list[itr])
                        if stem_pattern[-1] == " ":
                            stem_pattern = stem_pattern[:-1]

                        stems.append(stem_pattern)

            subject = GoogleTranslator(source="en", target=language[0]).translate(
                dataset[i]["subject"]
            )
            try:
                pd_df_dict["subject"].append(subject)
            except KeyError:
                pd_df_dict["subject"] = [subject]

            object = GoogleTranslator(source="en", target=language[0]).translate(
                dataset[i]["object"]
            )
            try:
                pd_df_dict["object"].append(object)
            except KeyError:
                pd_df_dict["object"] = [object]

            try:
                pd_df_dict["false"].append(counterfact_save_list)
            except KeyError:
                pd_df_dict["false"] = [counterfact_save_list]
            # if all the stems are the same, just save that one
            if len(set(stems)) == 1:
                try:
                    pd_df_dict["stem"].append(stems[0])
                except KeyError:
                    pd_df_dict["stem"] = [stems[0]]
            else:
                try:
                    pd_df_dict["stem"].append(str(stems))
                except KeyError:
                    pd_df_dict["stem"] = [str(stems)]
            same_stem = True

            if same_stem:
                # print(" ".join(translated_true_tokenized[:-1]))
                pass
            else:
                stems = []
                # print(stems)

            try:
                pd_df_dict["dataset_id"].append(dataset[i]["dataset_id"])
            except KeyError:
                pd_df_dict["dataset_id"] = [dataset[i]["dataset_id"]]
            try:
                pd_df_dict["relation"].append(dataset[i]["relation"])
            except KeyError:
                pd_df_dict["relation"] = [dataset[i]["relation"]]

        # print(pd_df_dict)
        df = pd.DataFrame.from_dict(pd_df_dict)
        df = df[
            ["dataset_id", "stem", "true", "false", "relation", "subject", "object"]
        ]
        for itr in range(len(df)):
            if len(df.loc[itr, "false"]) == 0:
                df.drop(itr, inplace=True)
        df.dropna(inplace=True)
        print(df)
        df.to_csv("test_translation.csv", index=False)

        # Optionally upload final csv to HuggingFace
        if args.hugging_face:
            data_files = {
                "English": "../../data/ingested_data/fact-checking-full-input-information-3-21-23.csv",
                "French": "test_translation.csv",
            }
            dataset = load_dataset("csv", data_files=data_files)

            # This reads the environment variables inside .env
            load_dotenv()
            # Logs into HF hub
            login(os.getenv("HF_TOKEN"))
            # push to hub
            dataset.push_to_hub("CalibraGPT/Fact_Checking")
            # test loading from hub
            load_dataset("CalibraGPT/Fact_Checking")


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
