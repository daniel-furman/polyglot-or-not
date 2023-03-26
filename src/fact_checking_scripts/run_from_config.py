"""
Main script for running fact checking with contrastive knowledge assessment

Instead of running the full benchmark, this script enables users to enter a
custom config file for their experiment. For example, the demo notebook that
tests a handful of example facts uses this function.
"""

from argparse import ArgumentParser
import importlib
from transformers import set_seed

from compare_models import compare_models


def main(config):
    set_seed(42)

    score_dicts = compare_models(
        config["models"], config["input_information"], config["verbosity"]
    )

    return score_dicts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("configs", type=str, help="Config file to set up run cka")
    args = parser.parse_args()
    config = importlib.import_module(args.configs).config
    score_dicts = main(config)
    print(f"\nScore dict summary:\n{score_dicts[1]}")
