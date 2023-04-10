"""
Fact Checking Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking dataset
Original sources cited in the project's README

Example usage:
python cache_fact_completion_dataset.py
"""

import os
import json
import pandas as pd
import numpy as np
import copy
from argparse import ArgumentParser
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv


def main(args):
    print("Caching the fact_completion dataset...")

    # Load calinet data
    with open(
        "../../data/calinet_data_original/probing-data-trex-500each.json", "r"
    ) as f:
        data_calinet = json.load(f)
        starter_df = pd.DataFrame(list(data_calinet["data"]))

    # all of these have to do with fact id 1
    # the sentences are formed in this format...
    # the start of a factual sentence, involving the subject
    # and then two possibilities: one true and one false?
    # storing these, then, we should do something like
    # sentence stem | correct | incorrect
    # and we can strip out the <extra_id_x> parts
    # to keep it model agnostic
    starter_df["sentences"][0][0]

    # create containers to hold our clean data
    sentence_stems = []
    correct = []
    incorrect = []
    fact_ids = []
    relations = []
    subjects = []
    objects = []
    for index, row in starter_df.iterrows():
        sentence_list = row["sentences"]
        for entry in sentence_list:
            # minor cleanup
            cleaned_stem = entry[0].replace("<extra_id_0>", "[BLANK]").strip()
            cleaned_correct = (
                entry[1].replace("<extra_id_0>", "").replace("<extra_id_1>", "").strip()
            )
            cleaned_incorrect = (
                entry[2].replace("<extra_id_0>", "").replace("<extra_id_1>", "").strip()
            )

            # grab sub<->obj
            subjects_and_objects = pd.json_normalize(row["triplet"])
            subjects.append(subjects_and_objects.sub_label.values[0])
            objects.append(subjects_and_objects.obj_label.values[0])

            # commit
            sentence_stems.append(cleaned_stem)
            correct.append(cleaned_correct)
            incorrect.append(cleaned_incorrect)
            fact_ids.append(row["fact_id"])
            relations.append(row["relation"])

    # sanity check
    assert (
        len(sentence_stems)
        == len(correct)
        == len(incorrect)
        == len(fact_ids)
        == len(relations)
        == len(subjects)
        == len(objects)
    )

    # merge into big df
    trex_df = pd.DataFrame(
        {
            "fact_id": fact_ids,
            "relation": relations,
            "subject": subjects,
            "object": objects,
            "stem": sentence_stems,
            "true": correct,
            "false": incorrect,
        }
    )

    # how many stems end in [BLANK]? -> 50451, or about 1/3.
    c = 0
    for stem in trex_df["stem"]:
        if stem.endswith("[BLANK]."):
            c += 1

    print(
        f"\t- Calinet dataset: There are {c} valid stem/fact pairs in the source data."
    )
    print(
        f"\t- Calinet dataset: There are {len(trex_df)} valid counterfacts"
        " in the source data."
    )

    def check_for_causal_compatibility(stem):
        return stem.endswith("[BLANK].")

    def trim_stem(stem):
        if stem.endswith("[BLANK]."):
            return stem[0 : len(stem) - 9]

    trex_causal_df = trex_df[
        trex_df.apply(lambda x: check_for_causal_compatibility(x.stem), axis=1)
    ]
    trex_causal_df = trex_causal_df.copy()
    trimmed_stems = trex_causal_df.apply(lambda x: trim_stem(x.stem), axis=1)
    trex_causal_df["stem"] = list(trimmed_stems)

    unique_trex = (
        len(trex_causal_df["fact_id"].unique()) / len(trex_causal_df["fact_id"]) * 100
    )
    print(
        f"\t- Calinet dataset: Only about {np.round(unique_trex, decimals=1)}"
        "% of the stem/fact pairs are unique (many are paraphrased)."
    )

    # before sampling, attach arbitrary counter ID, to then track who gets removed
    trex_causal_df["calibra_id"] = range(50451)
    trex_causal_subset = (
        trex_causal_df.groupby("fact_id")
        .apply(lambda x: x.sample(1, random_state=42))
        .reset_index(drop=True)
    )
    assert trex_causal_subset.shape[0] == len(trex_causal_df["fact_id"].unique())
    trex_causal_subset.head()

    trex_causal_subset.tail()

    removed_ids = {}
    removed_counterfacts = {}
    for c_id in trex_causal_df["calibra_id"]:
        if c_id not in trex_causal_subset["calibra_id"].values:
            fact_id = trex_causal_df[trex_causal_df["calibra_id"] == c_id][
                "fact_id"
            ].values[0]
            counterfact = trex_causal_df[trex_causal_df["calibra_id"] == c_id][
                "false"
            ].values[0]
            removed_ids[str(c_id)] = int(fact_id)
            if str(fact_id) in removed_counterfacts:
                removed_counterfacts[str(fact_id)].append(counterfact)
            else:
                removed_counterfacts[str(fact_id)] = [counterfact]

    # did we remove as many rows as eq to the difference between the full calinet
    # dataset row number and the unique count?
    assert len(removed_ids) == trex_causal_df.shape[0] - len(
        trex_causal_df["fact_id"].unique()
    )
    # drop extraneous calibra_id column
    trex_causal_subset.drop(["calibra_id"], axis=1, inplace=True)
    # there are some fact_id's that only have 1 row
    # since we did pull stuff out based on our left to right requirement
    full_falses = {}
    for k, v in removed_counterfacts.items():
        subset_false = trex_causal_subset[
            trex_causal_subset["fact_id"] == int(k)
        ].false.values[0]
        full_falses[k] = v
        full_falses[k].append(subset_false)

    def replace_false_column(fact_id, false_val, full_false_dict=full_falses):
        if str(fact_id) in full_false_dict:
            return full_false_dict[str(fact_id)]
        else:
            return [false_val]

    replaced_falses = list(
        trex_causal_subset.apply(
            lambda x: replace_false_column(x.fact_id, x.false), axis=1
        )
    )

    trex_causal_subset["false"] = replaced_falses

    trex_causal_subset.head()
    trex_causal_subset.tail()
    data_calinet_input_information = {}
    trex_list = trex_causal_subset.to_dict("records")
    for i, entry in enumerate(trex_list):
        data_calinet_input_information[i] = trex_list[i]
    num_pairs = 0
    for x, y in data_calinet_input_information.items():
        data_calinet_input_information[x] = y
        data_calinet_input_information[x]["false"] = list(set(y["false"]))

        num_pairs += len(data_calinet_input_information[x]["false"])

    print(
        f"\t- Calinet dataset: There are {len(data_calinet_input_information)}"
        " stem/fact pairs left after removing paraphrases."
    )
    print(
        f"\t- Calinet dataset: There are {num_pairs} counterfacts left after"
        " removing paraphrases."
    )

    # Load in ROME counterfact data
    with open("../../data/rome_data_original/counterfact.json", "r") as f:
        data_rome = json.load(f)

    print(
        f"\n\t- Rome dataset: There are {len(data_rome)} valid stem/fact pairs"
        " in the source data."
    )
    print(
        f"\t- Rome dataset: There are {len(data_rome)} valid counterfacts in"
        " the source data."
    )
    data_rome_input_information = {}

    for i in range(len(data_rome)):
        stem = data_rome[i]["requested_rewrite"]["prompt"].replace(
            "{}", data_rome[i]["requested_rewrite"]["subject"]
        )

        data_rome_input_information[str(i)] = {
            "stem": stem,
            "true": data_rome[i]["requested_rewrite"]["target_true"]["str"],
            "false": [data_rome[i]["requested_rewrite"]["target_new"]["str"]],
            "case_id": data_rome[i]["case_id"],
        }
    # Combine the two datasets
    # start from a clean var

    data_rome = copy.deepcopy(data_rome_input_information)
    data_calinet = copy.deepcopy(data_calinet_input_information)
    mixed_itr = 0
    mixed_df = {}

    for x, y in data_calinet.items():
        y["dataset_original"] = "calinet_input_information"
        mixed_df[str(mixed_itr)] = y

        mixed_itr += 1

    for x, y in data_rome.items():
        y["dataset_original"] = "rome_counterfact_input_information"
        mixed_df[str(mixed_itr)] = y
        mixed_itr += 1

    # Convert dicts to pandas, then optionally upload to HuggingFace
    pairs_list = []
    for x, y in mixed_df.items():
        pairs = y["stem"] + " " + y["true"]
        pairs_list.append(pairs)

    print(
        f"\n\t- Combined dataset: There are {len(pairs_list)} stem/fact"
        " pairs after combining data."
    )

    pairs_list = []
    for x, y in mixed_df.items():
        for itr in range(len(y["false"])):
            pairs = y["stem"] + " " + y["true"] + " " + y["false"][itr]
            pairs_list.append(pairs)

    print(
        f"\t- Combined dataset: There are {len(pairs_list)} counterfacts"
        " after combining data."
    )

    # update mixed_df to have all info for rome then write that out.
    mixed_df = pd.DataFrame.from_dict(mixed_df).T
    # get rome info to look at:
    with open("../../data/rome_data_original/counterfact.json", "r") as f:
        data_rome_original = json.load(f)
        rome_df = pd.DataFrame.from_dict(data_rome_original)

    # 3/20 data frame cleanup: adding subject/object columns
    rome_subjects = {}
    rome_objects = {}
    rome_relations = {}

    for i, rewrite in enumerate(rome_df["requested_rewrite"]):
        rome_subjects[i] = rewrite["subject"]
        rome_objects[i] = rewrite["target_true"]["str"]
        rome_relations[i] = rewrite["relation_id"]

    assert (
        len(rome_subjects)
        == len(rome_objects)
        == len(rome_relations)
        == rome_df.shape[0]
    )
    subjects = []
    objects = []
    ids = []
    relations = []

    for row in mixed_df.iterrows():
        if row[1]["dataset_original"] == "calinet_input_information":
            subjects.append(row[1]["subject"])
            objects.append(row[1]["object"])
            relations.append(row[1]["relation"])
            ids.append("calinet_" + str(row[1]["fact_id"]))
        if row[1]["dataset_original"] == "rome_counterfact_input_information":
            # get case id
            case_id = row[1]["case_id"]

            # get subject
            subjects.append(rome_subjects[case_id])
            # get object
            objects.append(rome_objects[case_id])
            # get relation
            relations.append(rome_relations[case_id])
            ids.append("rome_" + str(case_id))

    assert len(subjects) == len(objects) == len(ids) == len(relations)
    mixed_df["subject"] = subjects
    mixed_df["object"] = objects
    mixed_df["relation"] = relations
    mixed_df["dataset_id"] = ids
    mixed_df.drop(["fact_id", "case_id", "dataset_original"], axis=1, inplace=True)
    assert not mixed_df.isnull().values.any()
    mixed_df.head()
    # re-arrange cols
    mixed_df = mixed_df[
        ["dataset_id", "stem", "true", "false", "relation", "subject", "object"]
    ]
    # write to file as .parquet
    mixed_df.to_parquet(
        "../../data/ingested_data/en-fact-completion-3-20-23.parquet",
        index=False,
    )

    # further cleanup based on an analysis of the top-100 most confident misses
    # delete erroneous entries in the dataset
    # (these were not exhaustively searched for, some
    # errors could still exist in the data)
    rows_to_delete = [
        "rome_19765",
        "calinet_9087",
        "rome_9674",
        "rome_13669",
        "rome_17792",
        "calinet_469",
        "calinet_12945",
        "rome_17452",
        "rome_597",
        "calinet_7656",
        "rome_16474",
        "rome_6020",
        "rome_9479",
        "calinet_5834",
        "rome_9414",
        "rome_6487",
        "rome_10852",
        "rome_14709",
        "rome_4358",
        "rome_10342",
        "calinet_12839",
        "rome_19963",
        "rome_5757",
        "rome_3604",
        "rome_8710",
        "calinet_2551",
        "rome_20688",
        "rome_15441",
        "calinet_12842",
        "calinet_9348",
        "calinet_2516",
        "calinet_12777",
        "rome_13682",
        "calinet_29",
        "calinet_3198",
        "rome_10178",
        "rome_19495",
        "rome_9674",
        "rome_13028",
        "calinet_5452",
        "rome_19963",
        "calinet_2568",
        "calinet_5475",
        "calinet_9555",
        "rome_19788",
        "rome_12483",
        "rome_14334",
        "calinet_10778",
        "rome_612",
        "rome_8416",
        "calinet_5133",
        "calinet_5185",
        "rome_1525",
        "rome_5796",
        "rome_1359",
        "rome_15982",
        "rome_12882",
        "rome_796",
        "rome_7201",
        "rome_4998",
        "calinet_9032",
        "rome_15759",
        "rome_8513",
        "rome_9528",
        "rome_9653",
        "rome_13961",
        "rome_14778",
        "rome_2140",
        "rome_16482",
        "rome_4091",
        "rome_11399",
        "rome_19798",
        "calinet_8491",
        "calinet_8312",
        "calinet_8413",
        "rome_11510",
        "calinet_1609",
        "calinet_10514",
        "calinet_8022",
        "calinet_3508",
        "calinet_10716",
        "calinet_10294",
        "calinet_5256",
        "calinet_11265",
        "calinet_11400",
        "calinet_3307",
        "rome_14732",
        "rome_2374",
        "rome_7730",
        "calinet_10137",
        "calinet_10391",
        "calinet_3722",
        "calinet_3613",
        "calinet_3132",
        "calinet_10574",
        "calinet_3306",
        "calinet_7200",
        "calinet_8310",
        "calinet_3199",
        "calinet_10171",
        "calinet_9368",
        "calinet_5324",
        "calinet_470",
        "calinet_9347",
        "calinet_10393",
        "calinet_9445",
        "rome_10198",
        "calinet_5669",
        "rome_7352",
        "rome_1814",
        "calinet_5334",
        "rome_16980",
        "calinet_12130",
        "calinet_494",
        "calinet_1878",
        "calinet_3864",
        "rome_9081",
        "calinet_5849",
        "calinet_8111",
        "rome_8201",
        "calinet_4579",
        "calinet_10145",
        "calinet_1637",
        "calinet_5803",
        "rome_626",
        "calinet_9319",
        "calinet_5982",
        "calinet_6694",
        "calinet_6537",
        "calinet_5736",
        "calinet_1522",
        "calinet_5551",
        "calinet_10221",
        "calinet_3969",
        "calinet_6304",
        "calinet_3549",
        "calinet_1829",
        "calinet_3544",
        "calinet_8465",
        "calinet_6629",
        "calinet_12082",
        "calinet_1819",
        "rome_9477",
        "calinet_10184",
        "calinet_5905",
        "calinet_5671",
        "rome_15012",
        "rome_398",
        "calinet_12611",
        "calinet_609",
        "rome_3106",
        "rome_18739",
        "rome_3929",
        "calinet_4504",
        "calinet_2829",
        "calinet_2263",
        "calinet_6596",
        "calinet_7812",
        "calinet_12256",
        "calinet_5838",
        "calinet_1577",
        "calinet_1538",
        "rome_4716",
        "rome_7858",
        "calinet_12153",
        "calinet_8452",
        "rome_19436",
        "calinet_8223",
        "rome_9317",
        "calinet_9578",
        "calinet_1602",
        "calinet_3377",
        "calinet_7072",
        "calinet_8153",
        "calinet_2832",
        "calinet_7417",
        "calinet_7676",
        "calinet_10130",
        "calinet_8450",
        "calinet_7898",
        "calinet_9660",
        "calinet_10233",
        "rome_1131",
        "calinet_8450",
        "rome_18347",
        "rome_17012",
        "calinet_3302",
        "rome_2809",
        "calinet_5713",
        "rome_12017",
        "calinet_9441",
        "calinet_5750",
        "calinet_3636",
        "calinet_12289",
        "calinet_3556",
        "calinet_3589",
        "calinet_3523",
        "calinet_2807",
        "calinet_6282",
        "calinet_3605",
        "calinet_104",
        "calinet_447",
        "calinet_3947",
        "calinet_3966",
        "calinet_12194",
        "calinet_401",
        "rome_3244",
        "calinet_6969",
        "rome_5017",
        "calinet_2379",
        "calinet_2063",
        "calinet_2140",
        "calinet_9994",
        "calinet_2418",
        "calinet_2084",
        "calinet_1941",
        "rome_4233",
        "calinet_7183",
        "calinet_12628",
        "calinet_9160",
        "calinet_12789",
        "rome_1906",
        "calinet_137",
        "calinet_4516",
        "rome_19266",
        "rome_15136",
        "rome_15088",
        "calinet_5977",
        "calinet_3324",
        "calinet_8503",
    ]

    # delete these rows
    for i in list(mixed_df.index):
        if mixed_df.loc[i].dataset_id in list(set(rows_to_delete)):
            # print(mixed_df.loc[i].dataset_id)
            mixed_df.drop(i, axis=0, inplace=True)

    print(
        f"\t- Combined dataset: Removed {len(set(rows_to_delete))}"
        " stem/fact pairs that were manually flagged as errors."
    )

    mixed_df.shape[0]

    # delete stems that end with "a" or "an"
    itr = 0
    for i in list(mixed_df.index):
        if (mixed_df.loc[i].stem[-2:] == " a") or (mixed_df.loc[i].stem[-3:] == " an"):
            itr += 1
            mixed_df.drop(i, axis=0, inplace=True)

    print(f'\t- Combined dataset: Removed {itr} stem/fact pairs with "a/an + _".')
    mixed_df.reset_index(drop=True, inplace=True)

    # modify errors when sandwitched with correct data where possible
    # dictionary: dataset_id and new counterfact list, with the error removed
    rows_to_alter = {
        "calinet_7809": {"false": ["Gaulish", "Georgian"]},
        "calinet_1917": {"false": ["theology", "free software", "accounting"]},
        "calinet_7790": {"false": ["Hebrew", "Swahili"]},
        "rome_11311": {"false": ["Russian"], "true": "French", "object": "French"},
        "rome_17917": {"false": ["French"], "true": "Russian", "object": "Russian"},
        "calinet_7612": {"false": ["Phir Subah Hogi"]},
        "calinet_12317": {
            "false": ["Emperor", "Prime Minister"],
            "true": "President",
            "object": "President",
        },
        "rome_5908": {"false": ["violin"], "true": "guitar", "object": "guitar"},
        "rome_21907": {"false": ["French"], "true": "English", "object": "English"},
        "calinet_11761": {"false": ["Apple"]},
        "calinet_2821": {"stem": "The Italian capital is", "subject": "capital"},
        "calinet_10786": {
            "false": ["America", "Germany"],
            "true": "Russia",
            "object": "Russia",
        },
        "calinet_12059": {
            "false": ["President", "Supreme Court Justice"],
        },
        "calinet_143": {
            "false": ["California", "Canada"],
        },
        "calinet_5721": {
            "stem": "New Delhi is the capital city of",
            "true": "India",
            "false": ["Pakistan"],
            "subject": "New Delhi",
        },
        "calinet_12403": {
            "true": "Prime Minister",
            "false": ["President", "Governor"],
            "object": "Prime Minister",
        },
        "rome_18212": {
            "stem": "The Australian Open is held in",
            "false": ["Sydney"],
            "subject": "The Australian Open",
        },
        "rome_19787": {
            "stem": "The 1993 Bombay bombings took place in",
            "false": ["New Delhi"],
        },
        "calinet_6228": {
            "stem": "Marvin Gaye passed way in",
            "true": "Los Angeles",
            "false": ["Houston"],
            "object": "Los Angeles",
        },
        "calinet_681": {
            "true": "Athens",
            "false": ["Sparta", "Corinth"],
            "object": "Athens",
        },
        "calinet_7198": {
            "true": "Molsheim, France",
            "object": "Molsheim, France",
            "false": ["Maranello, Italy"],
        },
        "rome_17865": {
            "stem": "What does Wanda Sykes do? They write",
            "subject": "Wanda Sykes",
            "false": ["literature"],
        },
        "calinet_3768": {
            "stem": "Pearl Jam was formed in",
            "true": "Seattle",
            "object": "Seattle",
            "false": ["Los Angeles"],
        },
        "calinet_9216": {
            "stem": "Rolling Stone Magazine is written in",
            "subject": "Rolling Stone Magazine",
            "false": ["Spanish"],
        },
        "calinet_5824": {
            "stem": "Hungary, which has the capital",
            "object": "Hungary",
            "false": ["Vienna"],
        },
        "calinet_12363": {
            "false": ["Senator"],
            "true": "President",
            "object": "President",
        },
        "calinet_2180": {
            "false": ["Munich, Denver, Boston"],
        },
        "calinet_2742": {
            "true": "Munich",
            "false": ["Prague"],
            "object": "Munich",
        },
        "calinet_2820": {
            "false": ["Kampala"],
        },
        "calinet_8922": {
            "false": ["Honda"],
        },
        "calinet_5926": {
            "false": ["Zhejiang Province"],
        },
        "calinet_10906": {
            "false": ["Canada"],
        },
        "calinet_10852": {"false": ["Canada"], "true": "America", "object": "America"},
        "calinet_5749": {
            "true": "Lebanon",
            "false": ["Syria"],
            "object": "Lebanon",
        },
        "calinet_2811": {
            "stem": "The capital city of America is",
            "subject": "America",
        },
        "calinet_10312": {
            "stem": "America is affiliated with",
            "false": ["Warsaw Pact"],
            "subject": "America",
        },
        "calinet_5576": {
            "false": ["Jamaica"],
        },
        "calinet_6356": {
            "false": ["Philadelphia"],
        },
        "calinet_156": {
            "false": ["Uganda"],
        },
        "calinet_8171": {
            "false": ["Taylor Swift"],
        },
        "calinet_7004": {
            "false": ["Seattle, Washington"],
        },
        "calinet_5516": {
            "false": ["Norway"],
        },
        "calinet_8388": {
            "false": ["hip hop"],
        },
        "rome_9037": {
            "stem": "How I Met Your Mother is a",
        },
        "calinet_12121": {
            "stem": "Lyndon Johnson, who has the position of",
            "false": ["Vice President", "UN Secretary-General"],
        },
        "calinet_6983": {
            "false": ["New York City, New York"],
        },
        "calinet_8410": {"stem": "Metal musicians", "false": ["Coldplay"]},
        "rome_15790": {
            "true": "astronomy",
            "object": "astronomy",
            "false": ["literature"],
        },
        "calinet_1797": {
            "false": ["anthropology"],
        },
        "rome_17056": {
            "false": ["literature"],
        },
        "calinet_3106": {
            "false": ["Pakistan"],
        },
        "calinet_5819": {
            "false": ["Vancouver", "Quebec City", "Toronto"],
        },
        "calinet_6739": {
            "false": ["London"],
            "true": "Cambridge",
            "object": "Cambridge",
        },
        "calinet_2640": {"false": ["Kyoto"], "true": "Tokyo", "object": "Tokyo"},
        "rome_11966": {
            "false": ["Budapest"],
            "stem": "Austria, which has the capital",
            "subject": "Austria",
        },
        "calinet_7356": {
            "false": ["Rome"],
        },
        "calinet_209": {
            "false": ["Thailand"],
        },
        "calinet_42": {
            "false": ["Texas"],
        },
        "calinet_7125": {
            "false": ["Dallas", "San Francisco", "Chicago"],
        },
        "rome_9881": {
            "true": "folk",
            "object": "folk",
        },
        "rome_9646": {
            "true": "Basel",
            "object": "Basel",
        },
    }

    for key, dictionary in rows_to_alter.items():
        for column, edit in dictionary.items():
            row_ind = mixed_df[mixed_df.dataset_id == key].false.index[0]
            mixed_df.loc[row_ind, column] = edit

    # fix small syntax and grammatical errors, remove templates scheduled to be dropped
    for i in range(len(mixed_df)):
        # bespoke syntax fixes
        if "shares border with" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "shares border with", "shares a border with"
            )
        elif "shares the border with" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "shares the border with", "shares a border with"
            )
        elif "borders with" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "borders with", "shares a border with"
            )
        if "premiered" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "premiered", "originally aired"
            )
        if "The Smashing Pumpkins, who plays" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "The Smashing Pumpkins, who plays", "The Smashing Pumpkins, who play"
            )
        if "is to debut on" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "is to debut on", "originally aired on"
            )
        if mixed_df.loc[i].stem.split(" ")[-1] == "debuted":
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem + " on"

    # remove rows where the true answer is in the stem
    itr_true_in_stem = 0
    for i in range(len(mixed_df)):
        delete_bool = False
        for word in mixed_df.loc[i].true.lower().split(" "):
            if mixed_df.loc[i].stem.lower().count(word) > 0:
                delete_bool = True
        if delete_bool:
            itr_true_in_stem += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_true_in_stem} stem/fact pairs"
        " where the fact is explicitly stated in the stem"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove sister city related relations
    itr_sister_city = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P190":
            itr_sister_city += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_sister_city} stem/fact pairs"
        " that were relation P190 (sister city)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove religion related rows
    itr_religion = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P140":
            itr_religion += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_religion} stem/fact pairs"
        " that were relation P140 (religion)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove "tie diplomatic ties" items
    itr_diplomatic = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P530":
            itr_diplomatic += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_diplomatic} stem/fact pairs"
        " that were relation P530 (diplomatic ties)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove "citizen of" items
    itr_citizen = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P27":
            itr_citizen += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_citizen} stem/fact pairs that"
        " were relation P27 (citizen of)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove "affiliated with" items
    itr_affiliated = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P463":
            itr_affiliated += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_affiliated} stem/fact pairs"
        " that were relation P463 (affiliated with)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove soccer/football comparisons
    itr_football_soccer = 0
    for i in range(len(mixed_df)):
        if (mixed_df.loc[i].true == "soccer") or (mixed_df.loc[i].true == "football"):
            if ("soccer" in mixed_df.loc[i].false) or (
                "football" in mixed_df.loc[i].false
            ):
                if len(mixed_df.loc[i].false) == 1:
                    itr_football_soccer += 1
                    mixed_df.drop(index=i, inplace=True)
                else:
                    if "soccer" in mixed_df.loc[i].false:
                        false_list = copy.deepcopy(mixed_df.loc[i].false)
                        false_list.remove("soccer")
                        mixed_df.loc[i, "false"] = false_list
                    if "football" in mixed_df.loc[i].false:
                        false_list = copy.deepcopy(mixed_df.loc[i].false)
                        false_list.remove("football")
                        mixed_df.loc[i, "false"] = false_list
    print(
        f"\t- Combined dataset: Removed {itr_football_soccer} stem/fact pairs"
        " that compared football with soccer"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove rows with "expired at"
    itr_expired = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].stem.lower().count("expired at") > 0:
            itr_expired += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_expired} stem/fact pairs with"
        ' "expired at" wording'
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove stems / true / false elements with "-language" in them
    itr_dash_language = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].true.lower().count("-language") > 0:
            itr_dash_language += 1
            mixed_df.drop(index=i, inplace=True)
        elif mixed_df.loc[i].stem.lower().count("-language") > 0:
            itr_dash_language += 1
            mixed_df.drop(index=i, inplace=True)
        else:
            false_list = copy.deepcopy(mixed_df.loc[i].false)
            for false in mixed_df.loc[i].false:
                if false.lower().count("-language") > 0:
                    false_list.remove(false)
            if len(false_list) > 0:
                mixed_df.loc[i, "false"] = false_list
            else:
                itr_dash_language += 1
                mixed_df.drop(index=i, inplace=True)
    print(
        f"\t- Combined dataset: Removed {itr_dash_language} stem/fact pairs"
        ' with "-language" wording'
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove true / false elements that start with "the"
    itr_the = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].true.lower().split(" ")[0] == "the":
            itr_the += 1
            mixed_df.drop(index=i, inplace=True)
        else:
            false_list = copy.deepcopy(mixed_df.loc[i].false)
            for false in mixed_df.loc[i].false:
                if false.lower().split(" ")[0] == "the":
                    false_list.remove(false)
            if len(false_list) > 0:
                mixed_df.loc[i, "false"] = false_list
            else:
                itr_the += 1
                mixed_df.drop(index=i, inplace=True)
    print(
        f"\t- Combined dataset: Removed {itr_the} stem/fact pairs with"
        ' facts/counterfacts starting with "the"'
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # find any duplicates resulting from above fixes
    # start with [stem + fact] pairs
    pairs_list = []
    pairs_list_duplicated = []
    itrs_duplicated = []
    for i in range(len(mixed_df)):
        pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true)
        if pairs in pairs_list:
            pairs_list_duplicated.append(pairs)
            itrs_duplicated.append(i)
        pairs_list.append(pairs)

    print(
        f"\t- Combined dataset: Removed {len(pairs_list) - len(set(pairs_list))}"
        " stem/fact pair duplicates."
    )

    # repair any duplicates resulting from above fixes
    pairs_list_collect = []
    for i in range(len(mixed_df)):
        pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true)
        if pairs in pairs_list_duplicated:
            pairs_list_collect.append(
                (mixed_df.loc[i].stem, mixed_df.loc[i].true, mixed_df.loc[i].false)
            )
    new_counterfacts = {}
    for element in pairs_list_collect:
        try:
            new_counterfacts[element[0] + " " + element[1]].extend(element[2])
        except KeyError:
            new_counterfacts[element[0] + " " + element[1]] = element[2]
    new_counterfacts_2 = {}
    for x, y in new_counterfacts.items():
        new_counterfacts_2[x] = list(set(y))
    for i in range(len(mixed_df)):
        key_item = mixed_df.loc[i].stem + " " + mixed_df.loc[i].true
        if key_item in list(new_counterfacts_2.keys()):
            mixed_df.loc[i, "false"] = new_counterfacts_2[key_item]
    mixed_df.drop_duplicates(subset=["stem", "true"], inplace=True)
    mixed_df.reset_index(drop=True, inplace=True)

    # check duplicates were removed correctly
    pairs_list = []
    for i in range(len(mixed_df)):
        pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true)
        pairs_list.append(pairs)

    # shuffle the df's rows (without replacement)
    mixed_df = mixed_df.sample(
        frac=1, replace=False, random_state=44, ignore_index=True
    )

    # grab a subsest to include at the head, for sharing purposes
    good_subset = [
        "rome_21844",
        "rome_9500",
        "rome_9881",
        "rome_11754",
        "calinet_8922",
        "rome_9037",
        "calinet_2820",
        "rome_10452",
        "rome_5025",
        "rome_15553",
        "rome_13484",
        "rome_957",
        "rome_14462",
        "rome_20584",
        "rome_11479",
        "calinet_5926",
        "rome_1397",
        "calinet_12363",
        "rome_21333",
        "rome_8738",
        "calinet_5824",
        "rome_8783",
        "calinet_12059",
        "calinet_4311",
        "calinet_143",
        "rome_3074",
        "rome_20293",
        "calinet_12403",
        "rome_1437",
        "calinet_4036",
        "rome_12802",
        "rome_15752",
        "rome_19787",
        "calinet_6228",
        "calinet_2742",
        "rome_15619",
        "calinet_681",
        "calinet_7198",
        "rome_17865",
        "calinet_3768",
        "calinet_9216",
        "calinet_12590",
        "calinet_5749",
        "calinet_10312",
        "calinet_6356",
        "calinet_5576",
        "rome_18369",
        "calinet_156",
        "rome_5419",
        "calinet_8171",
        "rome_17929",
        "calinet_7004",
        "calinet_5516",
        "rome_14610",
        "calinet_8388",
        "rome_20965",
        "rome_10068",
        "rome_11693",
        "calinet_5684",
        "rome_3161",
        "calinet_12121",
        "calinet_6266",
        "rome_16867",
        "calinet_8410",
        "rome_4790",
        "rome_15892",
        "calinet_10027",
        "rome_3586",
        "rome_259",
        "rome_10548",
        "rome_10145",
        "rome_15334",
        "calinet_4853",
        "calinet_7490",
        "calinet_7109",
        "rome_21525",
        "rome_2650",
        "rome_4897",
        "calinet_1797",
        "rome_17056",
        "rome_1633",
        "rome_6657",
        "rome_21008",
        "calinet_5819",
        "calinet_6739",
        "rome_17595",
        "rome_20749",
        "rome_7651",
        "calinet_5813",
        "calinet_7356",
        "calinet_209",
        "rome_1077",
        "rome_21420",
        "calinet_42",
        "rome_15949",
        "rome_9753",
        "calinet_7125",
        "rome_20823",
        "rome_6678",
        "rome_9989",
        "rome_997",
        "rome_9925",
        "rome_991",
        "rome_9836",
        "rome_9740",
        "rome_9717",
        "rome_9707",
        "rome_9646",
        "rome_964",
        "rome_9628",
        "rome_9606",
    ]
    good_subset.reverse()
    for dataset_id in good_subset:
        id = mixed_df[mixed_df.dataset_id == dataset_id].index
        mixed_df = pd.concat([mixed_df.loc[id], mixed_df])
    mixed_df.drop_duplicates(subset=["dataset_id"], inplace=True)
    mixed_df.dropna(inplace=True)
    mixed_df.reset_index(drop=True, inplace=True)
    # make sure all counterfacts are sets
    pairs_list = []
    for i in range(len(mixed_df)):
        mixed_df.loc[i, "false"] = list(set(mixed_df.loc[i, "false"]))

    # find any duplicates remaining
    # there shouldn't be, after the set command above
    pairs_list = []
    for i in range(len(mixed_df)):
        pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true)
        pairs_list.append(pairs)
    print(
        f"\t- Combined dataset: There are {len(set(pairs_list))} unique stem/fact pairs"
        ' remaining in the final "CalibraGPT/Fact_Checking" dataset.'
    )
    assert len(set(pairs_list)) == len(mixed_df)

    pairs_list = []
    for i in range(len(mixed_df)):
        for item in mixed_df.loc[i].false:
            pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true, item)
            pairs_list.append(pairs)

    print(
        f"\t- Combined dataset: There are {len(set(pairs_list))} unique counterfacts"
        ' remaining in the final "CalibraGPT/Fact_Checking" dataset.'
    )

    # convert lists to strings with <br> delimiters
    for i in range(len(mixed_df)):
        if len(mixed_df.loc[i].false) == 1:
            mixed_df.loc[i, "false"] = mixed_df.loc[i, "false"][0]
        else:
            string = mixed_df.loc[i, "false"][0]
            for element in mixed_df.loc[i, "false"][1:]:
                string += " <br> " + element
            mixed_df.loc[i, "false"] = string

    # capitalize the first letter
    for i in range(len(mixed_df)):
        stem = mixed_df.loc[i].stem[0].capitalize() + mixed_df.loc[i].stem[1:]
        mixed_df.loc[i, "stem"] = stem

    # write to file as .parquet
    mixed_df.to_parquet(
        "../../data/ingested_data/en-fact-completion-3-21-23.parquet",
        index=False,
    )

    # order by language popularity

    # Optionally upload final parquet to HuggingFace
    if args.hugging_face:
        data_files = {
            "English": "../../data/ingested_data/en-fact-completion-3-21-23.parquet",
            "Spanish": "../../data/ingested_data/translated_versions/es-fact-completion-4-8-23.parquet",
            "French": "../../data/ingested_data/translated_versions/fr-fact-completion-4-5-23.parquet",
            "Russian": "../../data/ingested_data/translated_versions/ru-fact-completion-4-7-23.parquet",
            "Portuguese": "../../data/ingested_data/translated_versions/pt-fact-completion-4-8-23.parquet",
            "German": "../../data/ingested_data/translated_versions/de-fact-completion-4-7-23.parquet",
            "Italian": "../../data/ingested_data/translated_versions/it-fact-completion-4-9-23.parquet",
            "Ukrainian": "../../data/ingested_data/translated_versions/uk-fact-completion-4-9-23.parquet",
            "Romanian": "../../data/ingested_data/translated_versions/ro-fact-completion-4-5-23.parquet",
            "Czech": "../../data/ingested_data/translated_versions/cs-fact-completion-4-10-23.parquet",
            "Bulgarian": "../../data/ingested_data/translated_versions/bg-fact-completion-4-10-23.parquet",
            "Serbian": "../../data/ingested_data/translated_versions/sr-fact-completion-4-5-23.parquet",
            "Hungarian": "../../data/ingested_data/translated_versions/hu-fact-completion-4-5-23.parquet",
            "Croatian": "../../data/ingested_data/translated_versions/hr-fact-completion-4-5-23.parquet",
            "Danish": "../../data/ingested_data/translated_versions/da-fact-completion-4-5-23.parquet",
            "Slovenian": "../../data/ingested_data/translated_versions/sl-fact-completion-4-5-23.parquet",
            "Polish": "../../data/ingested_data/translated_versions/pl-fact-completion-4-5-23.parquet",
            "Dutch": "../../data/ingested_data/translated_versions/nl-fact-completion-4-5-23.parquet",
            "Catalan": "../../data/ingested_data/translated_versions/ca-fact-completion-4-5-23.parquet",
            "Swedish": "../../data/ingested_data/translated_versions/sv-fact-completion-4-5-23.parquet",
        }
        dataset = load_dataset("parquet", data_files=data_files)

        # This reads the environment variables inside .env
        load_dotenv()
        # Logs into HF hub
        login(os.getenv("HF_TOKEN"))
        # push to hub
        dataset.push_to_hub("CalibraGPT/Fact-Completion")
        # test loading from hub
        load_dataset("CalibraGPT/Fact-Completion")

    return None


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
