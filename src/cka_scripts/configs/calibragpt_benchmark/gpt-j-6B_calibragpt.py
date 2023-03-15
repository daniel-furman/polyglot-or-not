import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        "EleutherAI/gpt-j-6B",
        # "togethercomputer/GPT-JT-6B-v1"
    ],
    "input_information": input_info,
    "verbosity": False,
}
