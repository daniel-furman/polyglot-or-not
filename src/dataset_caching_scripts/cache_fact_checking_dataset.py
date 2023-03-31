"""
Fact Checking Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking dataset
Original sources cited in the project's README

Example usage:
python cache_fact_checking_dataset.py
"""

import os
import json
import pandas as pd
import copy
from argparse import ArgumentParser
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv


def main(args):
    print("Caching the fact_checking dataset...")

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
        f"\t- Calinet dataset: There are {len(trex_df)} valid counterfacts in the source data."
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

    print(
        f'\t- Calinet dataset: Only about {str(len(trex_causal_df["fact_id"].unique())/len(trex_causal_df["fact_id"])*100)[0:4]}% of the stem/fact pairs are unique (many are paraphrased).'
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

    # did we remove as many rows as eq to the difference between the full calinet dataset row number and the unique count?
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
        f"\t- Calinet dataset: There are {len(data_calinet_input_information)} stem/fact pairs left after removing paraphrases."
    )
    print(
        f"\t- Calinet dataset: There are {num_pairs} counterfacts left after removing paraphrases."
    )
    # out of curiosity, which relation templates persist in the cleaned, 'causal friendly' set...
    # print(trex_causal_df["relation"].value_counts())

    # Load in ROME counterfact data
    with open("../../data/rome_data_original/counterfact.json", "r") as f:
        data_rome = json.load(f)

    print(
        f"\n\t- Rome dataset: There are {len(data_rome)} valid stem/fact pairs in the source data."
    )
    print(
        f"\t- Rome dataset: There are {len(data_rome)} valid counterfacts in the source data."
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
        f"\n\t- Combined dataset: There are {len(pairs_list)} stem/fact pairs after combining data."
    )

    pairs_list = []
    for x, y in mixed_df.items():
        for itr in range(len(y["false"])):
            pairs = y["stem"] + " " + y["true"] + " " + y["false"][itr]
            pairs_list.append(pairs)

    print(
        f"\t- Combined dataset: There are {len(pairs_list)} counterfacts after combining data."
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
    # write to file as .csv
    mixed_df.to_csv(
        "../../data/ingested_data/fact-checking-full-input-information-3-20-23.csv",
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
    ]

    # delete these rows
    for i in list(mixed_df.index):
        if mixed_df.loc[i].dataset_id in list(set(rows_to_delete)):
            # print(mixed_df.loc[i].dataset_id)
            mixed_df.drop(i, axis=0, inplace=True)

    print(
        f"\t- Combined dataset: Removed {len(set(rows_to_delete))} rows that were manually flagged as errors."
    )

    mixed_df.shape[0]
    # delete stems that end with "a" or "an"
    itr = 0
    for i in list(mixed_df.index):
        if (mixed_df.loc[i].stem[-2:] == " a") or (mixed_df.loc[i].stem[-3:] == " an"):
            itr += 1
            mixed_df.drop(i, axis=0, inplace=True)

    print(
        f'\t- Combined dataset: Removed {itr} stem/fact pairs that contained "a/an + _" grammatical errors.'
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # modify errors when sandwitched with correct data where possible
    # dictionary: dataset_id and new counterfact list, with the error removed
    rows_to_alter = {
        "calinet_7809": {"false": "['Gaulish', 'Georgian']"},
        "calinet_1917": {"false": "['theology', 'free software', 'accounting']"},
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
        "calinet_11761": {"false": "['Apple']"},
        "calinet_2821": {"stem": "The Italian capital is", "subject": "capital"},
    }

    for key, dictionary in rows_to_alter.items():
        for column, edit in dictionary.items():
            row_ind = mixed_df[mixed_df.dataset_id == key].false.index[0]
            mixed_df.loc[row_ind, column] = edit

    # fix small syntax and grammatical errors, remove templates scheduled to be dropped

    itr_religion = 0
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
        elif "premiered" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "premiered", "originally aired"
            )
        elif "The Smashing Pumpkins, who plays" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "The Smashing Pumpkins, who plays", "The Smashing Pumpkins, who play"
            )
        elif "is to debut on" in mixed_df.loc[i].stem:
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem.replace(
                "is to debut on", "originally aired on"
            )
        elif mixed_df.loc[i].stem.split(" ")[-1] == "debuted":
            mixed_df.loc[i, "stem"] = mixed_df.loc[i].stem + " on"

    # remove religion related rows
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P140":
            itr_religion += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_religion} stem/fact pairs that were relation P140 (religion related)"
    )
    mixed_df.reset_index(drop=True, inplace=True)

    # remove "tie diplomatic ties" items
    itr_diplomatic = 0
    for i in range(len(mixed_df)):
        if mixed_df.loc[i].relation == "P530":
            itr_diplomatic += 1
            mixed_df.drop(index=i, inplace=True)

    print(
        f"\t- Combined dataset: Removed {itr_diplomatic} stem/fact pairs that were relation P530 (diplomatic relations)"
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
        f"\t- Combined dataset: Removed {itr_football_soccer} stem/fact pairs that compared football with soccer"
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
        f"\t- Combined dataset: Removed {len(pairs_list) - len(set(pairs_list))} stem/fact pair duplicates."
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

    # make sure all counterfacts are sets
    pairs_list = []
    for i in range(len(mixed_df)):
        mixed_df.loc[i, "false"] = list(set(mixed_df.loc[i, "false"]))

    # shuffle the df's rows (without replacement)
    mixed_df = mixed_df.sample(
        frac=1, replace=False, random_state=44, ignore_index=True
    )

    # grab a subsest to include at the head, for sharing purposes
    good_subset = [
        "rome_11754",
        "calinet_10852",
        "calinet_8922",
        "calinet_2820",
        "rome_10452",
        "rome_5025",
        "rome_15553",
        "rome_13484",
        "rome_20283",
        "rome_957",
        "rome_15088",
        "rome_14462",
        "rome_20584",
        "rome_11479",
        "calinet_10906",
    ]
    good_subset.reverse()
    for dataset_id in good_subset:
        id = mixed_df[mixed_df.dataset_id == dataset_id].index
        mixed_df = pd.concat([mixed_df.loc[id], mixed_df])
    mixed_df.drop_duplicates(subset=["dataset_id"], inplace=True)
    mixed_df.reset_index(drop=True, inplace=True)

    # find any trio duplicates remaining (there shouldn't be, after the set command above)
    pairs_list = []
    for i in range(len(mixed_df)):
        pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true)
        pairs_list.append(pairs)
    print(
        f'\t- Combined dataset: There are {len(set(pairs_list))} unique stem/fact pairs remaining in the final "CalibraGPT/Fact_Checking" dataset.'
    )
    assert len(set(pairs_list)) == len(mixed_df)

    pairs_list = []
    for i in range(len(mixed_df)):
        for item in mixed_df.loc[i].false:
            pairs = (mixed_df.loc[i].stem, mixed_df.loc[i].true, item)
            pairs_list.append(pairs)

    print(
        f'\t- Combined dataset: There are {len(set(pairs_list))} unique counterfacts remaining in the final "CalibraGPT/Fact_Checking" dataset.'
    )

    # write to file as .csv
    mixed_df.to_csv(
        "../../data/ingested_data/fact-checking-full-input-information-3-21-23.csv",
        index=False,
    )

    # Optionally upload final csv to HuggingFace
    if args.hugging_face:
        data_files = {
            "English": "../../data/ingested_data/fact-checking-full-input-information-3-21-23.csv",
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
