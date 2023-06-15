import os
import glob
import torch
from argparse import Namespace
from transformers import set_seed
from datasets import load_dataset

from compare_models import compare_models

# args config for running the benchmark
args = Namespace(
    model="tiiuae/falcon-40b",
    language="en",
)

print(args)

# ensure GPU access
if not torch.cuda.is_available():
    raise Exception("Change runtime type to include a GPU.")

# set warning level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# run the fact completion benchmark
print("Running the fact_completion benchmark...")

set_seed(42)

# change the below to work on a list of ("english", "en") pairs
# load in the dataset corresponding to the input language
supported_languages = [
    ("english", "en"),
    ("french", "fr"),
    ("spanish", "es"),
    ("german", "de"),
    ("ukrainian", "uk"),
    ("romanian", "ro"),
    ("bulgarian", "bg"),
    ("catalan", "ca"),
    ("danish", "da"),
    ("croatian", "hr"),
    ("hungarian", "hu"),
    ("italian", "it"),
    ("dutch", "nl"),
    ("polish", "pl"),
    ("portuguese", "pt"),
    ("russian", "ru"),
    ("slovenian", "sl"),
    ("serbian", "sr"),
    ("swedish", "sv"),
    ("czech", "cs"),
]

dataset_bool = False
for lang_arr in supported_languages:
    if (args.language.lower() == lang_arr[0]) or (args.language.lower() == lang_arr[1]):
        dataset = load_dataset(
            "Polyglot-or-Not/Fact-Completion",
            split=lang_arr[0].capitalize(),
        )
        dataset_bool = True

if not dataset_bool:
    raise Exception("Language not supported.")

# check the input model is compatible
compatible_model_prefixes = [
    "t5",
    "pythia",
    "gpt",
    "opt",
    "llama",
    "roberta",
    "bert",
    "bloom",
    "stablelm",
    "mpt",
    "redpajama",
    "falcon",
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

# run the contrastive knowledge assessment function
# logs saved at './content/logging/'
score_dicts, log_fpath = compare_models(
    config["models"], config["input_information"], config["verbosity"]
)

# print the summary results
print(f"\nScore dict summary:\n{score_dicts[1]}")


# save result logs to drive

print(log_fpath)
