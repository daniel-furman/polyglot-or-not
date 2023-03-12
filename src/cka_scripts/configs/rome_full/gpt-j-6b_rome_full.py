import json

with open("../../data/rome_counterfact_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        "EleutherAI/gpt-j-6B",
    ],
    "input_information": input_info,
    "verbosity": False,
}
