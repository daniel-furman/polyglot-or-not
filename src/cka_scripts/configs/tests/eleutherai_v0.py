config = {
    "models": [
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3b",
        #"EleutherAI/gpt-neo-2.7b",
        #"EleutherAI/gpt-j-6B",
        #"EleutherAI/gpt-neox-20B",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        #"EleutherAI/pythia-2.8b",
        #"EleutherAI/pythia-6.9b",
        #"EleutherAI/pythia-12b",
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
