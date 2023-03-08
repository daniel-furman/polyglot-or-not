config = {
    "models": [
        "google/flan-t5-small",  # 80M params
        "google/flan-t5-base",  # 250M params
        "google/flan-t5-large",  # 780M params
        # "google/flan-t5-xl",  # 3B params
        # "google/flan-t5-xxl",  # 11B params
        # "google/flan-ul2",  # 20B params
    ],
    "input_information": {
        "The 2020 Olympics were held in": ["Tokyo", "Berlin"],
        "Operation Overlord took place in": ["Normandy", "Manila"],
        "Steve Jobs is the founder of": ["Apple", "Microsoft"],
    },
    "verbosity": False,
}
