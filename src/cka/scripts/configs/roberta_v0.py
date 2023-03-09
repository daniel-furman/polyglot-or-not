config = {
    "models": [
        "distilroberta-base",  # 82M params
        "roberta-base",  # 125M params
        "xlm-roberta-base",  # 125M params
        "roberta-large",  # 354M params
        "xlm-roberta-large",  # 354M params
    ],
    "input_information": {
        "The 2020 Olympics were held in": ["Tokyo", "London"],
        "Operation Overlord took place in": ["Normandy", "Manila"],
        "Steve Jobs is the founder of": ["Apple", "Microsoft"],
    },
    "verbosity": False,
}
