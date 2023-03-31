"""
Main script for running fact checking with contrastive knowledge assessment

Runs the full CalibraGPT/Fact_Checking benchmark dataset

Example usage:
python main.py --model distilgpt2 --language en
"""

from argparse import ArgumentParser
from transformers import set_seed
from datasets import load_dataset

from compare_models import compare_models


def main(args):
    print("Running the fact_checking benchmark...")

    set_seed(42)

    # load in the dataset corresponding to the input language
    if (args.language.lower() == "english") or (args.language.lower() == "en"):
        dataset = load_dataset("CalibraGPT/Fact_Checking", split="English")
    else:
        raise Exception("Language not supported.")

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
        raise Exception("Model not supported.")

    # create a config for running the pipeline
    config = {
        "models": [args.model],
        "input_information": dataset,
        "verbosity": True,
    }

    print(config)
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
        help='Name of the hugging face model to run (e.g., "distilbert-base-uncased").',
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="english",
        help='Name of the language to run (e.g., "english" or "en").',
    )

    args = parser.parse_args()
    main(args)
