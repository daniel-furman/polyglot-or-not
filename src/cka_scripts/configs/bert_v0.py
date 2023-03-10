config = {
    "models": [
        "google/bert_uncased_L-2_H-128_A-2",  # 4.4M params (Bert-Tiny)
        "google/bert_uncased_L-4_H-256_A-4",  # 11.3M params (Bert-Mini)
        "google/bert_uncased_L-4_H-512_A-8",  # 29.1M params (Bert-Small)
        "google/bert_uncased_L-8_H-512_A-8",  # 41.7M params (Bert-Medium)
        "distilbert-base-uncased",  # 66M params
        "bert-base-uncased",  # 110M params
        "bert-large-uncased",  # 330M params
    ],
    "input_information": {
        "The 2020 Olympics were held in": ["Tokyo", "London"],
        "Operation Overlord took place in": ["Normandy", "Manila"],
        "Steve Jobs is the founder of": ["Apple", "Microsoft"],
    },
    "verbosity": False,
}
