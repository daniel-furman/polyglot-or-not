"""
Language Translation Dataset Caching Script

Run this script to re-produce caching the CalibraGPT/Fact_Checking
dataset's Non-English splits
Original sources cited in the project's README

Example usage:
python cache_multilingual_fact_checking_dataset.py --hugging_face False
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
    print("Translating the fact_checking dataset into Non-English languages...")

    dataset = load_dataset("CalibraGPT/Fact_Checking")
    print(dataset)

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