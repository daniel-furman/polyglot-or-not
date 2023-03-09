config = {
    "models": [
        "distilgpt2",  # 82M params
        "gpt2",  # 124M params
        "gpt2-medium",  # 355M params
        "gpt2-large",  # 774M params
        # "gpt2-xl",  # 1.5B params
    ],
    "input_information": {
        "The 2020 Olympics were held in": ["Tokyo", "London"],
        "Operation Overlord took place in": ["Normandy", "Manila"],
        "Steve Jobs is the founder of": ["Apple", "Microsoft"],
    },
    "verbosity": False,
}
