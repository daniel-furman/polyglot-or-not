import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from probe_helpers import probe_flan, probe_gpt2, probe_bert

device = torch.device("cuda")


# first, write helper to pull a pretrained LM and tokenizer off the shelf
def get_model_and_tokenizer(model_name):
    if "flan" in model_name.lower():
        return T5Tokenizer.from_pretrained(
            model_name
        ), T5ForConditionalGeneration.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto"
        )

    elif "gpt" in model_name.lower():
        return AutoTokenizer.from_pretrained(
            model_name
        ), AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, device_map="auto"
        )

    elif "bert" in model_name.lower():
        return AutoTokenizer.from_pretrained(
            model_name
        ), AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(
            device
        )


# next, write a helper to pull a probe function for the given LM
def get_probe_function(prefix):
    probe_functions = [probe_flan, probe_gpt2, probe_bert]
    for func in probe_functions:
        if prefix.lower() in func.__name__:
            return func


# lastly, write a wrapper function to compare models
def compare_models(model_name_list, input_pairings, verbose):

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
    score_dict_succinct = {}

    for model_name in model_name_list:
        print(f"CKA for {model_name}")
        print("Loading  model...")

        # get proper model and tokenizer
        tokenizer, model = get_model_and_tokenizer(model_name)

        print("Running comparisons...")

        # establish prefix
        prefix = ""
        probe_func = None

        # get correct CKA function
        if ("t5" in model_name.lower()) or ("ul2" in model_name.lower()):
            prefix = "flan"
            probe_func = get_probe_function(prefix)

        elif "gpt" in model_name.lower():
            prefix = "gpt"
            probe_func = get_probe_function(prefix)

        elif "roberta" in model_name.lower():
            prefix = "roberta"
            probe_func = get_probe_function("bert")

        elif "bert" in model_name.lower():
            prefix = "bert"
            probe_func = get_probe_function(prefix)

        # iterate over context/entity pairings
        # input_pairings is a dict
        # context is a plain string (since our context's will be unique)
        # and entities is a list containing, in the first slot, the true
        # value for the statement and in the subsequent slots, incorrect information

        for context, entities in input_pairings.items():
            entity_count = 0
            p_true = 0.0
            p_false = 0.0

            if prefix == "flan":
                context += " <extra_id_0>."
            elif prefix == "roberta":
                context += " <mask>."
            elif prefix == "bert":
                context += " [MASK]."

            for entity in entities:
                target_id = None
                # first find target vocab id
                # default to the very first token that get's predicted
                # e.g. in the case of Tokyo, which gets split into <Tok> <yo>,

                if prefix == "flan":
                    target_id = tokenizer.encode(
                        entity,
                        padding="longest",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)[0][0]

                elif prefix == "gpt":
                    target_id = tokenizer.encode(" " + entity, return_tensors="pt").to(
                        device
                    )[0][0]

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

                # next call probe function
                model_prob = probe_func(model, tokenizer, target_id, context, verbose)

                # lastly, register results
                if entity_count == 0:
                    p_true = model_prob

                else:
                    p_false += model_prob

                entity_count += 1

            p_false /= entity_count - 1

            try:
                score_dict_full[model_name.lower()].append(
                    {
                        context: {
                            "p_true": p_true,
                            "p_false": p_false,
                            "p_true - p_false": p_true - p_false,
                            "p_true > p_false": p_true > p_false,
                        }
                    }
                )
            except KeyError:
                score_dict_full[model_name.lower()] = [
                    {
                        context: {
                            "p_true": p_true,
                            "p_false": p_false,
                            "p_true - p_false": p_true - p_false,
                            "p_true > p_false": p_true > p_false,
                        }
                    }
                ]

            try:
                score_dict_succinct[model_name.lower()].append(
                    {
                        context: {
                            "p_true > p_false": p_true > p_false,
                        }
                    }
                )
            except KeyError:
                score_dict_succinct[model_name.lower()] = [
                    {
                        context: {
                            "p_true > p_false": p_true > p_false,
                        }
                    }
                ]

        print("Done\n")
        del tokenizer
        del model
        torch.cuda.empty_cache()

    return score_dict_full, score_dict_succinct
