import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": ["bigscience/bloom-7b1", "bigscience/bloom-3b"],
    "input_information": input_info,
    "verbosity": False,
}
