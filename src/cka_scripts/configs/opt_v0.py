config = {
    "models": [
        "facebook/opt-125m",  # 125M params
        "facebook/opt-350m",  # 350M params
        "facebook/opt-1.3b",  # 1.3b params
        "facebook/opt-2.7b",  # 2.7b params
        "facebook/opt-6.7b",  # 6.7b params
        "facebook/opt-13b",  # 13b params
    ],
    "input_information": {
        "0": {
            "stem": "The 2020 Olympics were held in",
            "true": "Tokyo",
            "false":["London","Berlin", "Chicago"]
        },
        "1": {
            "stem": "Operation Overlord took place in",
            "true": "Normandy",
            "false":["Manila","Santiago", "Baghdad"]
        },
        "2": {
            "stem": "Steve Jobs is the founder of",
            "true": "Apple",
            "false":["Microsoft","Oracle", "Intel"]
        }
    },
    "verbosity": False,
}
