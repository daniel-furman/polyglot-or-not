config = {
    "models": [
        "distilroberta-base",  # 82M params
        "roberta-base",  # 125M params
        "xlm-roberta-base",  # 125M params
        "roberta-large",  # 354M params
        "xlm-roberta-large",  # 354M params
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
