from argparse import ArgumentParser
import importlib

from transformers import set_seed

from compare_models import compare_models


def main(config):

    set_seed(42)

    score_dict_full, score_dict_succinct = compare_models(
        config["models"], config["input_information"]
    )

    return score_dict_full, score_dict_succinct


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("configs", type=str, help="Config file to set up run pipeline")
    args = parser.parse_args()
    config = importlib.import_module(args.configs).config
    score_dict_full, score_dict_succinct = main(config)
    print(f"\nScore dict full:\n{score_dict_full}")
    print(f"\nScore dict succinct:\n{score_dict_succinct}")
