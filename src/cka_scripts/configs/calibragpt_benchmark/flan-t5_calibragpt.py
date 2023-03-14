import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        # "google/flan-t5-small",  # 80M params
        # "google/flan-t5-base",  # 250M params
        # "google/flan-t5-large",  # 780M params
        # "google/flan-t5-xl",  # 3B params
        "google/flan-t5-xxl",  # 11B params
    ],
    "input_information": input_info,
    "verbosity": False,
}
