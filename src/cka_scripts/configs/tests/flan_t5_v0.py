config = {
    "models": [
        "google/flan-t5-small",  # 80M params
        "google/flan-t5-base",  # 250M params
        "google/flan-t5-large",  # 780M params
        # "google/flan-t5-xl",  # 3B params
        # "google/flan-t5-xxl",  # 11B params
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
