from argparse import ArgumentParser
import json
from transformers import set_seed

from compare_models import compare_models


def main(args):

    set_seed(42)

    # change data loading from json to Hugging Face dataset
    # load in the dataset for the input language
    if args.language == "english":
        with open("../../data/calibragpt_full_input_information.json", "r") as f:
            input_info = json.load(f)
    else:
        raise Exception("Language not supported")

    # check the input model is compatible
    compatible_model_prefixes = [
        "flan",
        "t5",
        "pythia",
        "gpt",
        "opt",
        "llama",
        "roberta",
        "bert",
        "bloom",
    ]

    model_supported = False
    for model_prefix in compatible_model_prefixes:
        if model_prefix in args.model.lower():
            model_supported = True

    if not model_supported:
        raise Exception(
            "Model not supported yet, create an issue / PR to add a new HF compatible model"
        )

    # create a config for running the pipeline
    config = {
        "models": [args.model],
        "input_information": input_info,
        "verbosity": False,
    }

    # run the contrastive knowledge assessment function
    # logs saved at './content/logging/'
    score_dicts = compare_models(
        config["models"], config["input_information"], config["verbosity"]
    )

    # print the summary results
    print(f"\nScore dict summary:\n{score_dicts[1]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-small",
    )
    parser.add_argument("--language", "-l", type=str, default="english")
    args = parser.parse_args()
    main(args)
