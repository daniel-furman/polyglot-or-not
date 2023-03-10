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
