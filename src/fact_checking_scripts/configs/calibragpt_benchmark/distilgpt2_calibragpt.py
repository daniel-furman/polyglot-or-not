import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        "distilgpt2",
    ],
    "input_information": input_info,
    "verbosity": False,
}
