config = {
    "models": [
        "distilgpt2",  # 82M params
        "gpt2",  # 124M params
        "gpt2-medium",  # 355M params
        "gpt2-large",  # 774M params
        "gpt2-xl",  # 1.5B params
    ],
    "input_information": {
        "0": {
            "stem": "The 2020 Olympics were held in",
            "true": "Tokyo",
            "false": ["London", "Berlin", "Chicago"],
        },
        "1": {
            "stem": "Operation Overlord took place in",
            "true": "Normandy",
            "false": ["Manila", "Santiago", "Baghdad"],
        },
        "2": {
            "stem": "Steve Jobs is the founder of",
            "true": "Apple",
            "false": ["Microsoft", "Oracle", "Intel"],
        },
    },
    "verbosity": False,
}
