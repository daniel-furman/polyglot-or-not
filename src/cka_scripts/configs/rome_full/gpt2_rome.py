import json

with open("../../../../data/rome_counterfact_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        # "distilgpt2",  # 82M params
        # "gpt2",  # 124M params
        # "gpt2-medium",  # 355M params
        "gpt2-large",  # 774M params
        # "gpt2-xl",  # 1.5B params
    ],
    "input_information": input_info,
    "verbosity": False,
}
