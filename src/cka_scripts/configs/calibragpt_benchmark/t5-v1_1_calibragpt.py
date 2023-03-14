import json

with open("../../data/calibragpt_full_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        # "google/t5-v1_1-small",  # 80M params
        # "google/t5-v1_1-base",  # 250M params
        # "google/t5-v1_1-large",  # 780M params
        "google/t5-v1_1-xl",  # 3B params
        # "google/t5-v1_1-xxl",  # 11B params
    ],
    "input_information": input_info,
    "verbosity": False,
}
