# Polyglot or Not?: Measuring Multilingual Encyclopedic Knowledge Retrieval from Foundation Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This is the repository for the following paper: [Polyglot or Not?: Measuring Multilingual Encyclopedic Knowledge Retrieval from Foundation Language Models](https://arxiv.org/abs/2305.13675). It contains several research artifacts, including:

1. The [code][cka_run_main] for running the fact-completion test
2. Our [dataset][hf_data] of factual associations translated into 20 languages
3. A [demo][cka_lightweight_demo] of contrastive knowledge assessment 

## Method

Given a factual association such as *The capital of France is **Paris***, we determine whether a model adequately "knows" the correct completion with the following test:
 
* **Step 1**: prompt the model to predict the likelihood of the token **Paris** following *The Capital of France is*
* **Step 2**: prompt the model to predict the average likelihood of a set of false, counterfactual tokens following the same stem.
 
If the value from **Step 1** is greater than the value from **Step 2** we conclude that the model adequately recalls that fact. Formally, this is an application of the *Contrastive Knowledge Assessment* proposed in [[1][bib]]. 

## Models Evaluated

We evaluate 5 open-sourced foundation models of interest, like [LLaMa](https://arxiv.org/abs/2302.13971) [[2][bib]]. We perform this assessment using 303k fact-completions translated into 20 languages ([results](https://github.com/daniel-furman/Polyglot-or-Not#test-results)). 

In addition to our multilingual assessment, we also scored 20 models (like [GPT-NeoX](https://arxiv.org/abs/2204.06745) and [OPT](https://arxiv.org/abs/2205.01068)) on the English-only subset of our dataset. 

While we would have liked to test close-sourced models, such as OpenAI's GPT-4, such models don't provide vocabulary-wide token probabilities at inference and are thus incompatible with our test. 

## Data Release

We present 303k unique fact-completions in [`Polyglot-or-Not/Fact-Completion`][hf_data], which are in the form of {stem, fact, counterfact} triples. See the [dataset viewer](https://huggingface.co/datasets/Polyglot-or-Not/Fact-Completion/viewer/Polyglot-or-Not--Fact-Completion/English) for a closer look. 

* 20 Latin/Cyrillic script languages are included. The ISO 639-1 language codes are: `bg`, `ca`, `cs`, `da`, `de`, `en`, `es`, `fr`, `hr`, `hu`, `it`, `nl`, `pl`, `pt`, `ro`, `ru`, `sl`, `sr`, `sv`, and `uk`. 

The factual associations were originally sourced from English-language Wikidata curated in the T-REx dataset [[3][bib]] as utilized in factual association research such as [[1][bib]] and [[4][bib]]. We used the Google Translate API alongside bespoke wrapper [code](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/src/dataset_caching_scripts/language_translation_helper.py) to programmatically generate the non-English cuts. 

## Test Results 

 ### **Multilingual** leaderboard
 
 | model            | accuracy (%)   | params | *n* tokens
 |------------------|:--------------:|:--------------:|:--------------:|
 | [llama-33b](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) | **79.31** (+/- 0.74) | 32.5B | 1.4T |
 | [m-bert](https://huggingface.co/bert-base-multilingual-cased) |  **62.00** (+/- 0.87) | 110M | - |
 | [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)  | **57.70** (+/- 0.88) | 7.1B | 341B |
 | [xlm-roberta](https://huggingface.co/xlm-roberta-large) | **56.03** (+/- 0.90) | 355M | 295B |
 | [mt5-xl](https://huggingface.co/google/mt5-xl) |  **52.51** (+/- 0.91) | 3.7B | - |
 | Random Baseline | 50 | - | - |

 **Table 1**: Multilingual test leaderboard. Here, **accuracy** refers to the average performance of each model across 20 distinct languages. The uncertainty estimates represent averaged 95% confidence intervals computed from 10000 bootstrap iterations per language. **Params** and ***n* tokens** record each model’s number of parameters and number of dataset tokens, respectively (when such data is available). These results reveal that models struggle to recall facts in a multilingual setting, as compared to their English-only performance (Table [2](https://github.com/daniel-furman/Polyglot-or-Not#english-only-leaderboard)). For instance, on average, LLaMA-33B's accuracy decreased by approximately 11% from English to non-English languages.

 &nbsp; 

 ### **English-only** leaderboard
 
 | model            | accuracy (%)   | params | *n* tokens
 |------------------|:--------------:|:--------------:|:--------------:|
 | [llama-33b](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) | **89.40** (+/- 0.38) | 32.5B | 1.4T |
 | [llama-13b](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) | **86.66** (+/- 0.42) | 12.5B | 1T |
 | [llama-7b](https://huggingface.co/docs/transformers/main/model_doc/llama#llama) | **85.53** (+/- 0.43) | 6.7B | 1T |
 | [redpajama-7b](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-7B-v0.1) | **85.07** (+/- 0.44) | 7B | 800B |
 | [mpt-7b](https://huggingface.co/mosaicml/mpt-7b) | **83.39** (+/- 0.46) | 7B | 1T |
 | [opt-30b](https://huggingface.co/facebook/opt-30b) | **82.09** (+/- 0.47) | 30B | 180B |
 | [redpajama-3b](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) | **82.09** (+/- 0.47) | 3B | 800B |
 | [opt-13b](https://huggingface.co/facebook/opt-13b) | **81.94** (+/- 0.46) | 13B | 30B |
 | [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) | **81.50** (+/- 0.47) | 20B | 420B |
 | [gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) |  **81.14** (+/- 0.47) | 6B | 420B |
 | [pythia-12b](https://huggingface.co/EleutherAI/pythia-12b) | **80.53** (+/- 0.48) | 12B | 420B |
 | [t5-v1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | **76.55** (+/- 0.52) | 13B | 34B |
 | [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) | **76.16** (+/- 0.51) | 7.1B | 341B |
 | [gpt2-xl](https://huggingface.co/gpt2-xl) | **73.76** (+/- 0.54) | 1.5B | - |
 | [bert](https://huggingface.co/bert-base-uncased) | **72.60** (+/- 0.54) | 110M | - |
 | [m-bert](https://huggingface.co/bert-base-multilingual-cased) | **71.80** (+/- 0.55) | 110M | - |
 | [stablelm-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b) | **68.85** (+/- 0.55) | 7B | 1.5T |
 | [mt5-xxl](https://huggingface.co/google/mt5-xxl) | **61.58** (+/- 0.59) | 13B | - |
 | [xlm-roberta](https://huggingface.co/xlm-roberta-large) | **61.55** (+/- 0.59) | 355M | 295B |
 | [mt5-xl](https://huggingface.co/google/mt5-xl) |  **59.96** (+/- 0.59) | 3.7B | - |
 | Random Baseline | 50   | - | - |
 
 **Table 2**: Monolingual test leaderboard. **Accuracy** represents performance on English-only data. The uncertainty estimates are 95% confidence intervals computed from 10000 bootstrap iterations. **Params** and ***n* tokens** record each model’s number of parameters and number of dataset tokens, respectively (when such data is available. Consistent with the trends in Table [1](https://github.com/daniel-furman/Polyglot-or-Not#multilingual-leaderboard), LLaMAs of varying sizes emerge as the front-runners.
 
 &nbsp;

### **LLaMA-33B** performance across languages

![LLaMA test leaderboard](notebooks/viz/assets/LLaMa_h_bar_plot_final.png)

**Figure 1**: LLaMA-33B's test performance across languages. Accuracy denotes the model's performance assessed individually for each language. The LLaMA-33B model demonstrates higher proficiency with languages utilizing the Latin script as compared to those using the Cyrillic script (Ukrainian, Bulgarian, Russian, and Serbian). A [chi-squared test](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/notebooks/error_analysis/EntitySigTesting.ipynb) substantiates a significant dependency of the model's test performance on the language script (*χ2* = 3570.576, *p* < 0.001).

## Authors

* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>
* Shreshta Bhat <bhat_shreshta@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Citation

Please cite this repository as follows if you use its data or code:

```
@misc{polyglot_or_not,
  author = {Daniel Furman and Tim Schott and Shreshta Bhat},
  title = {Polyglot or Not?: Measuring Multilingual Encyclopedic Knowledge Retrieval from Foundation Language Models},
  year = {2023}
  publisher = {GitHub},
  howpublished = {\url{https://github.com/daniel-furman/Polyglot-or-Not}},
}
```

## Bibliography 

[1] Calibrating Factual Knowledge in Pretrained Language Models. Dong, Qingxiu, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. In Findings of the Association for Computational Linguistics: EMNLP 2022. [arXiv:2210.03329][cka] (2022).

[2]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1 (2023).

* LLaMA weights were accessed with the approval of Meta AI and used in accordance with the License (see [link](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) for more details).  

[3] T-REx: A Large Scale Alignment of Natural Language with Knowledge Base Triples. ElSahar, Hady, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon S. Hare, Frédérique Laforest and Elena Paslaru Bontas Simperl. International Conference on Language Resources and Evaluation. [Link][trex] (2018).

[4] Mass Editing Memory in a Transformer. Meng, Kevin, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. arXiv preprint [arXiv:2210.07229][memit] (2022).


[bib]: https://github.com/daniel-furman/Polyglot-or-Not#bibliography
[hf_data]: https://huggingface.co/datasets/Polyglot-or-Not/Fact-Completion
[cka]: https://arxiv.org/abs/2210.03329
[memit]: https://arxiv.org/abs/2210.07229
[mmlu]: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
[mmlu_paper]: https://arxiv.org/abs/2009.03300
[trex]: http://aclanthology.lst.uni-saarland.de/L18-1544.pdf
[cka_lightweight_demo]: https://github.com/daniel-furman/Polyglot-or-Not/blob/main/notebooks/fact_completion_notebooks/fact-completion-lightweight-demo.ipynb
[cka_run_main]: https://github.com/daniel-furman/Polyglot-or-Not/blob/main/notebooks/fact_completion_notebooks/fact-completion-full-benchmark.ipynb
