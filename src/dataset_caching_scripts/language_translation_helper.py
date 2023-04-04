"""
Colab helper function for translating data into other languages
with Google Translate
"""

import re
import time
import pandas as pd
import tqdm

from datasets import load_dataset
from deep_translator import GoogleTranslator


def main(args):
    print("Translating the fact_checking dataset into Non-English languages...")

    languages = [args.language]

    # for each language, translate the dataset:
    for language in languages:
        pd_df_dict = {}
        dataset = load_dataset("CalibraGPT/Fact_Checking", split="English")
        if args.first_100:
            num_loops = 100
        else:
            num_loops = len(dataset)

        for i in tqdm.tqdm(range(num_loops)):
            try:
                # grab the stem + true fact to translate
                true_pair = dataset[i]["stem"] + " " + dataset[i]["true"]

                # grab all the stem + false facts to translate
                counterfacts_list = dataset[i]["false"].split(" <br> ")

                false_pair_list = []
                for counterfact in counterfacts_list:
                    false_pair_list.append(dataset[i]["stem"] + " " + counterfact)
                # translate the stem + true fact
                translated_true = GoogleTranslator(
                    source="en", target=language
                ).translate(true_pair)
                time.sleep(0.01)
                # translate the stem + false facts
                translated_false_list = []
                for false_pair in false_pair_list:
                    translated_false_list.append(
                        GoogleTranslator(source="en", target=language).translate(
                            false_pair
                        )
                    )
                    time.sleep(0.01)
                # word tokenize the translations
                translated_true_tokenized = translated_true.split(" ")
                translated_false_tokenized_list = []
                for itr, false_pair in enumerate(false_pair_list):
                    translated_false_tokenized_list.append(
                        translated_false_list[itr].split(" ")
                    )

                # grab values for all the different colunmns in fact_checking
                # grab the stems first
                stems = []
                # start with the true fact
                true_fact_translated = GoogleTranslator(
                    source="en", target=language
                ).translate(dataset[i]["true"])
                time.sleep(0.01)
                # see if the translated object is at the end of the sentence
                index_fact = len(translated_true_tokenized) - len(
                    true_fact_translated.split(" ")
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
                # otherwise, check if the original english object is at
                # the end of the sentence
                else:
                    index_fact = len(translated_true_tokenized) - len(
                        dataset[i]["true"].split(" ")
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

                # now add data for all the counterfacts
                try:
                    counterfact_save_list = []
                    for itr, counterfact in enumerate(counterfacts_list):
                        false_fact_translated = GoogleTranslator(
                            source="en", target=language
                        ).translate(counterfact)
                        time.sleep(0.01)
                        # see if the translated object is at the end of the sentence
                        index_fact = len(translated_false_tokenized_list[itr]) - len(
                            false_fact_translated.split(" ")
                        )
                        if (
                            false_fact_translated.lower()
                            in " ".join(
                                translated_false_tokenized_list[itr][index_fact:]
                            ).lower()
                        ):
                            counterfact_save_list.append(false_fact_translated)
                            stem_pattern = re.compile(
                                false_fact_translated, re.IGNORECASE
                            )
                            stem_pattern = stem_pattern.sub(
                                "", translated_false_list[itr]
                            )
                            if stem_pattern[-1] == " ":
                                stem_pattern = stem_pattern[0:-1]

                            stems.append(stem_pattern)

                        # otherwise, check if the original english object is at the
                        # end of the sentence
                        else:
                            index_fact = len(
                                translated_false_tokenized_list[itr]
                            ) - len(counterfacts_list[itr].split(" "))

                            if (
                                counterfacts_list[itr]
                                in " ".join(
                                    translated_false_tokenized_list[itr][index_fact:]
                                ).lower()
                            ):
                                counterfact_save_list.append(counterfacts_list[itr])
                                stem_pattern = re.compile(
                                    counterfacts_list[itr], re.IGNORECASE
                                )
                                stem_pattern = stem_pattern.sub(
                                    "", translated_false_list[itr]
                                )
                                if stem_pattern[-1] == " ":
                                    stem_pattern = stem_pattern[:-1]

                                stems.append(stem_pattern)

                except AttributeError:
                    pd_df_dict["true"].pop()
                    continue

                # add subject, object, and false fact list to the dataframe
                # check if the translated subject/object is in the sentence
                # otherwise, just use the original subject/object
                subject = GoogleTranslator(source="en", target=language).translate(
                    dataset[i]["subject"]
                )
                time.sleep(0.01)
                if dataset[i]["subject"] in translated_true:
                    try:
                        pd_df_dict["subject"].append(dataset[i]["subject"])
                    except KeyError:
                        pd_df_dict["subject"] = [dataset[i]["subject"]]
                else:
                    try:
                        pd_df_dict["subject"].append(subject)
                    except KeyError:
                        pd_df_dict["subject"] = [subject]

                object = GoogleTranslator(source="en", target=language).translate(
                    dataset[i]["object"]
                )
                time.sleep(0.01)
                if dataset[i]["object"] in translated_true:
                    try:
                        pd_df_dict["object"].append(dataset[i]["object"])
                    except KeyError:
                        pd_df_dict["object"] = [dataset[i]["object"]]
                else:
                    try:
                        pd_df_dict["object"].append(object)
                    except KeyError:
                        pd_df_dict["object"] = [object]

                # add the false fact list to the dataframe
                try:
                    pd_df_dict["false"].append(counterfact_save_list)
                except KeyError:
                    pd_df_dict["false"] = [counterfact_save_list]

                # if all the stems are the same, save the one stem to the dataframe
                if len(set(stems)) == 1:
                    try:
                        pd_df_dict["stem"].append(stems[0])
                    except KeyError:
                        pd_df_dict["stem"] = [stems[0]]

                # otherwise, save a list of n stems for n fact/counterfact completions
                else:
                    try:
                        pd_df_dict["stem"].append(stems)
                    except KeyError:
                        pd_df_dict["stem"] = [stems]

                # add dataset_id and relation
                try:
                    pd_df_dict["dataset_id"].append(dataset[i]["dataset_id"])
                except KeyError:
                    pd_df_dict["dataset_id"] = [dataset[i]["dataset_id"]]
                try:
                    pd_df_dict["relation"].append(dataset[i]["relation"])
                except KeyError:
                    pd_df_dict["relation"] = [dataset[i]["relation"]]

            except:
                print(f'ERROR: {dataset[i]["dataset_id"]}')
                print("\n")

        df = pd.DataFrame.from_dict(pd_df_dict)
        df = df[
            ["dataset_id", "stem", "true", "false", "relation", "subject", "object"]
        ]

        # drop the empty cells
        for itr in range(len(df)):
            if len(df.loc[itr, "false"]) == 0:
                df.drop(itr, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # convert lists to strings with <br> delimiters
        for i in range(len(df)):
            if len(df.loc[i].false) == 1:
                df.loc[i, "false"] = df.loc[i, "false"][0]
            else:
                string = df.loc[i, "false"][0]
                for element in df.loc[i, "false"][1:]:
                    string += " <br> " + element
                df.loc[i, "false"] = string
        for i in range(len(df)):
            if type(df.loc[i].stem) == list:
                string = df.loc[i, "stem"][0]
                for element in df.loc[i, "stem"][1:]:
                    string += " <br> " + element
                df.loc[i, "stem"] = string

        # remove " ." at end of stem if exists, loop through list stems when <br> exists
        # remove " " at the beginning and end of stems
        for i in range(len(df)):
            if df.loc[i].stem[-2:] == " .":
                df.loc[i, "stem"] = df.loc[i].stem[:-2]
            if " . <br> " in df.loc[i].stem:
                df.loc[i, "stem"] = df.loc[i].stem.replace(" . <br> ", " <br> ")
            if df.loc[i].stem[:1] == " ":
                df.loc[i, "stem"] = df.loc[i].stem[1:]
            if " <br>  " in df.loc[i].stem:
                df.loc[i, "stem"] = df.loc[i].stem.replace(" <br>  ", " <br> ")
            if "  <br> " in df.loc[i].stem:
                df.loc[i, "stem"] = df.loc[i].stem.replace("  <br> ", " <br> ")
            if "  <br>  " in df.loc[i].stem:
                df.loc[i, "stem"] = df.loc[i].stem.replace("  <br>  ", " <br> ")

        # capitalize the first letter
        for i in range(len(df)):
            stem = df.loc[i].stem[0].capitalize() + df.loc[i].stem[1:]
            df.loc[i, "stem"] = stem

        # check that the num of <br> chars are consistent between
        # stem and fact/counterfacts
        for i in range(len(df)):
            if " <br> " in df.loc[i].stem:
                if (
                    len(df.loc[i].stem.split(" <br> "))
                    != len(df.loc[i].false.split(" <br> ")) + 1
                ):
                    df.drop(i, inplace=True)

        df.reset_index(drop=True, inplace=True)

        # save to parquet
        df.to_parquet(
            f"/content/{args.language}-fact-checking-{args.datetime}.parquet",
            index=False,
        )
