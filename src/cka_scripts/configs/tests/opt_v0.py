config = {
    "models": [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        # "facebook/opt-2.7b",
        # "facebook/opt-6.7b",
        # "facebook/opt-13b",
        # "facebook/opt-30b",
        # "facebook/opt-66b",
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
