import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        # "distilroberta-base",  # 82M params
        # "roberta-base",  # 125M params
        # "xlm-roberta-base",  # 125M params
        "roberta-large",  # 354M params
        # "xlm-roberta-large",  # 354M params
    ],
    "input_information": input_info,
    "verbosity": False,
}
