import json

with open("../../data/rome_counterfact_input_information.json", "r") as f:
    input_info = json.load(f)

config = {
    "models": [
        "/content/drive/MyDrive/Colab Files/llama/LLaMA/int8/llama-30b/",
    ],
    "input_information": input_info,
    "verbosity": False,
}
