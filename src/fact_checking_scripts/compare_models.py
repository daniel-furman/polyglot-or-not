import datetime
import json
import os
import numpy as np
import tqdm
import torch

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from probe_helpers import probe_flan, probe_gpt, probe_bert, probe_llama, probe_t5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise Exception("Change runtime type to include a GPU.")


# first, write helper to pull a pretrained LM and tokenizer off the shelf
def get_model_and_tokenizer(model_name):
    if ("flan" in model_name.lower()) or "t5" in model_name.lower():
        return T5Tokenizer.from_pretrained(
            model_name
        ), T5ForConditionalGeneration.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
        )

    elif (
        ("gpt" in model_name.lower())
        or ("opt" in model_name.lower())
        or ("pythia" in model_name.lower())
        or ("bloom" in model_name.lower())
    ):
        return AutoTokenizer.from_pretrained(
            model_name
        ), AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
        )

    elif "bert" in model_name.lower():
        return AutoTokenizer.from_pretrained(
            model_name
        ), AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(
            device
        )

    elif "llama" in model_name.lower():
        # llama tokenizer path is expected to be one folder back from the input model
        # weights in a folder called "tokenizer"
        tokenizer_path = "/".join(model_name.split("/")[0:-1]) + "/tokenizer/"
        return transformers.LLaMATokenizer.from_pretrained(
            tokenizer_path
        ), transformers.LLaMAForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
        )


# next, write a helper to pull a probe function for the given LM
def get_probe_function(prefix):
    probe_functions = [probe_flan, probe_gpt, probe_bert, probe_llama, probe_t5]
    for func in probe_functions:
        if prefix.lower() in func.__name__:
            return func


# lastly, write a wrapper function to compare models
def compare_models(model_name_list, input_dataset, verbose):

    """
    Model-wise comparison helper function

    we should be able to do the following:
      * input a set of models we want to evaluate
      * input an expression of interest
      * input a 'true' next-token alonside a false
      * and get an output report that contains..
        * the 'result' ie is true > false
        * the probabilities of both of those values
      * running this method over a large set of positive/negative pairings should result in a large pool of information that can be used to compare model-families
      * we can also look at the relative 'certainty' across different models (at least in orders of magnitude)

    """

    score_dict_full = {}
    score_dict_summary = {}

    if not os.path.isdir("/content"):
        os.mkdir("/content")
    if not os.path.isdir("/content/logging"):
        os.mkdir("/content/logging")

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    for model_name in model_name_list:
        true_count = 0
        fact_count = 0
        p_falses = []
        p_trues = []

        print(f"CKA for {model_name}")
        print("Loading  model...")

        # get proper model and tokenizer
        tokenizer, model = get_model_and_tokenizer(model_name)

        print("Running comparisons...")

        # establish prefix
        prefix = ""
        probe_func = None

        # get correct CKA function
        if "flan" in model_name.lower():
            prefix = "flan"
            probe_func = get_probe_function(prefix)
        elif "t5" in model_name.lower():
            prefix = "t5"
            probe_func = get_probe_function(prefix)
        elif (
            ("gpt-neo" in model_name.lower())
            or ("gpt-j" in model_name.lower())
            or ("pythia" in model_name.lower())
        ):
            prefix = "eleutherai"
            probe_func = get_probe_function("gpt")

        elif "gpt" in model_name.lower():
            prefix = "gpt"
            probe_func = get_probe_function(prefix)

        elif "opt" in model_name.lower():
            prefix = "opt"
            probe_func = get_probe_function("gpt")

        elif "roberta" in model_name.lower():
            prefix = "roberta"
            probe_func = get_probe_function("bert")

        elif "bert" in model_name.lower():
            prefix = "bert"
            probe_func = get_probe_function(prefix)

        elif "llama" in model_name.lower():
            prefix = "llama"
            probe_func = get_probe_function(prefix)

        elif "bloom" in model_name.lower():
            prefix = "bloom"
            probe_func = get_probe_function("gpt")

        # iterate over context/entity pairings
        # input_dataset is a datasets dataset
        # context is a plain string (since our context's will be unique)
        # and entities is a list containing, in the first slot, the true
        # value for the statement and in the subsequent slots, incorrect information

        for entities_dict in tqdm.tqdm(input_dataset):
            # convert string of list into a real list
            counterfacts_list = (
                entities_dict["false"]
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(", ")
            )

            # intitiate vars
            p_true = 0.0
            p_false = 0.0
            p_false_list_inner = []

            # grab true and false entities
            entities = [entities_dict["true"]]
            entities.extend(counterfacts_list)

            # iterate through each fact and counterfact
            for entity_count, entity in enumerate(entities):

                # grab the context
                context = entities_dict["stem"]
                # if multiple stems are stored, grab the correct one
                # (zeroeth stem is true fact, next ones are counterfacts)
                if type(context) == list:
                    context = context[entity_count]
                # add necessary additions based on model type
                if prefix == "roberta":
                    context += " <mask>."
                elif prefix == "bert":
                    context += " [MASK]."
                elif prefix == "t5":
                    context += " <extra_id_0>."

                # first find target vocab id
                # default to the very first token that get's predicted
                # e.g. in the case of Tokyo, which gets split into <Tok> <yo>,
                target_id = None
                if (prefix == "flan") or (prefix == "t5"):
                    target_id = tokenizer.encode(
                        " " + entity,
                        padding="longest",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)[0][0]

                elif (
                    (prefix == "gpt") or (prefix == "eleutherai") or (prefix == "bloom")
                ):
                    target_id = tokenizer.encode(" " + entity, return_tensors="pt").to(
                        device
                    )[0][0]

                elif prefix == "opt":
                    target_id = tokenizer.encode(" " + entity, return_tensors="pt").to(
                        device
                    )[0][1]

                elif prefix == "roberta":
                    target_id = tokenizer.encode(
                        " " + entity,
                        padding="longest",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)[0][1]

                elif prefix == "bert":
                    target_id = tokenizer.encode(
                        entity,
                        padding="longest",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)[0][1]

                elif prefix == "llama":
                    target_id = tokenizer.encode(" " + entity, return_tensors="pt").to(
                        device
                    )[0][2]

                # next call probe function
                model_prob = probe_func(model, tokenizer, target_id, context, verbose)

                # lastly, register results
                # if it is the first time through, it is the fact
                if entity_count == 0:
                    p_true = model_prob
                # if it is the second+ time through, it is the counterfactual(s)
                else:
                    p_false += model_prob
                    p_false_list_inner.append(float(model_prob))

            # entity count is equal to the num counterfactuals
            # (since it started at a 0 index in the enumerate)
            p_false /= entity_count

            # record results:
            score_dict_full_data = {
                "stem": context,
                "fact": entities[0],
                "counterfact": entities[1:],
                "p_true": float(p_true),
                "p_false_list": p_false_list_inner,
                "p_false_average": float(p_false),
                "p_true / p_false_average": np.round(
                    float(p_true) / (float(p_false) + 1e-13), decimals=4
                ),
                "p_true > p_false_average": str(float(p_true) > float(p_false)),
            }

            # record the rest of the metadata
            score_dict_full_data["subject"] = entities_dict["subject"]
            score_dict_full_data["object"] = entities_dict["object"]
            score_dict_full_data["relation"] = entities_dict["relation"]
            score_dict_full_data["dataset_id"] = entities_dict["dataset_id"]

            # add results to the given model name
            try:
                score_dict_full[model_name.lower()].append(score_dict_full_data)
            except KeyError:
                score_dict_full[model_name.lower()] = [score_dict_full_data]

            # append p_false and p_true
            p_falses.append(float(p_false))
            p_trues.append(float(p_true))

            # update counts based on probs
            if p_true > p_false:
                true_count += 1
            fact_count += 1

        # record the summary dict
        score_dict_summary[
            model_name.lower()
        ] = f"This model predicted {true_count}/{fact_count} facts at a higher prob than the given counterfactual. The mean p_true was {np.round(np.mean(np.array(p_trues)), decimals=4)} while the mean p_false_average was {np.round(np.mean(np.array(p_falses)), decimals=4)}."

        print("Done\n")
        del tokenizer
        del model
        torch.cuda.empty_cache()

    score_dicts = [score_dict_full, score_dict_summary]

    # logging
    score_dicts_logging = {}
    score_dicts_logging["curr_datetime"] = str(now)
    try:
        score_dicts_logging["model_name"].append(model_name)
    except KeyError:
        score_dicts_logging["model_name"] = [model_name]

    score_dicts_logging["score_dict_summary"] = score_dict_summary
    score_dicts_logging["score_dict_full"] = score_dict_full

    with open(
        f"/content/logging/{prefix}_logged_cka_outputs_{dt_string}.json", "w"
    ) as outfile:
        json.dump(score_dicts_logging, outfile)

    return score_dicts
