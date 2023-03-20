from argparse import ArgumentParser
import json

from transformers import set_seed

from compare_models import compare_models


def main(args):

    set_seed(42)

    # change this to loading from hugging face dataset - language as split
    if args.language == "english":
        with open("../../data/calibragpt_full_input_information.json", "r") as f:
            input_info = json.load(f)
    else:
        raise Exception("Language not supported")

    config = {
        "models": [args.model],
        "input_information": input_info,
        "verbosity": False,
    }

    score_dicts = compare_models(
        config["models"], config["input_information"], config["verbosity"]
    )

    # add logic for saving logs to content/logging/<name>.json

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
