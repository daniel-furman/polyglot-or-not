# Polyglot or Not?: Measuring Multilingual Encyclopedic Knowledge Retrieval from Foundation Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Polyglot-or-Not/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This is the repository for [Polyglot or Not?: Measuring Multilingual Encyclopedic Knowledge Retrieval from Foundation Language Models](https://bit.ly/ischool-berkeley-capstone). It contains several research artifacts, including:

1. The [code][cka_run_main] for running the fact-completion test
2. Our release of the 20-language [test dataset][hf_data]
3. A [demo][cka_lightweight_demo] of contrastive knowledge assessment 

## Test Description

Given a factual association such as *The capital of France is **Paris***, we determine whether a model adequately "knows" the correct completion with the following test:
 
* **Step 1**: prompt the model to predict the likelihood of the token **Paris** following *The Capital of France is*
* **Step 2**: prompt the model to predict the average likelihood of a set of false, counterfactual tokens following the same stem.
 
If the value from **Step 1** is greater than the value from **Step 2** we conclude that the model adequately recalls that fact. Formally, this is an application of the Contrastive Knowledge Assessment proposed in [[1][bib]]. For every foundation model of interest, like [LLaMa](https://arxiv.org/abs/2302.13971) [[2][bib]], we perform this assessment on a set of facts translated into 20 languages. All told, we score foundation models on 303k fact-completions ([results](https://github.com/daniel-furman/Polyglot-or-Not#multilingual-fact-completion-results)). We also scored monolingual models (like [GPT-NeoX](https://arxiv.org/abs/2204.06745) and [OPT](https://arxiv.org/abs/2205.01068)) on the English-only subset.

## Data Release

We present 303k unique fact-completions in [`CalibraGPT/Fact-Completion.parquet`][hf_data], which are in the form of stem-fact-counterfact triples. See the [dataset viewer](https://huggingface.co/datasets/CalibraGPT/Fact-Completion/viewer/CalibraGPT--Fact-Completion/English) for a closer look. 

The factual associations were originally derived from Wikidata curated in the T-REx dataset [[3][bib]]. Since the T-REx dataset is English-only, we used the Google Translate API to programmatically generate the non-English cuts. We then had a handful of native speakers review small samples of the dataset for select languages to validate the faithfulness of the translations. Despite the positive sentiment from these reviews, minor language translation errors likely persist in small amounts within the dataset. 

* 20 Latin/Cyrillic script languages are included. The ISO 639-1 language codes are: bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, and uk. 

## Test Results 

&nbsp;

 **NB**: The bolded values in the tables indicate the percentage of fact completions adequately retrieved by the given model. The uncertainty estimates (+/-) represent 95% confidence intervals computed from 10000 bootstrap iterations.
 
 ### **Multilingual** fact-completion results.
 
 | Model            | 20 Languages<br />(avg % correct)      | Num Params | Authors      |  Org   |
 |------------------|:--------------:|:--------------:|--------------|--------------|
 | [llama-30b](https://arxiv.org/abs/2302.13971) | **79.31** +/- 0.74 | 30B | Touvron et al., 2023 | Meta | 
 | [m-bert-base](https://arxiv.org/abs/1810.04805) |  **62.00** +/- 0.87 | 110M | Devlin et al., 2018 | Google |
 | [bloom-7b1](https://arxiv.org/abs/2211.05100) | **57.70** +/- 0.88 | 7B | Scao et al., 2022 | BigScience |
 | [xlm-roberta-large](https://arxiv.org/abs/1911.02116) | **56.03** +/- 0.90 | 355M | Conneau et al., 2019 | Meta | 
 | [mt5-xl](https://arxiv.org/abs/2010.11934) |  **52.51** +/- 0.91 | 3.7B | Xue et al., 2020 | Google |
 | Random Baseline | 50 | &nbsp;| &nbsp; | &nbsp; |
 
 **Table 1**: Insert caption.

 &nbsp;

 ### **LLaMa-30B** multilingual performance.

![LLaMa test leaderboard](notebooks/viz/assets/LLaMa_h_bar_final.png)

**Figure 1**: The percentage of fact completions adequately retrieved by the LLaMa-30b model per language (blue). Overall, LLaMa-30b performs much better on Latin script compared to Cyrillic. A chi-square test was run with the *null* hypothesis that language script was independent from the LLaMa-30b performance, the results of which were statistically significant with *p* < 0.001.
 
 &nbsp;
 
 ### **English** fact-completion results.
 
 | Model            | English-only<br />(% correct)      | Num Params | Authors    |  Org   | 
 |------------------|:--------------:|:--------------:|--------------|--------------|
 | [llama-30b](https://arxiv.org/abs/2302.13971) | **89.40** +/- 0.38 |  30B |  Touvron et al., 2023 | Meta |
 | [llama-13b](https://arxiv.org/abs/2302.13971) | **86.66** +/- 0.42 |  13B |  Touvron et al., 2023 | Meta |
 | [llama-7b](https://arxiv.org/abs/2302.13971) | **85.53** +/- 0.43 |  7B |  Touvron et al., 2023 | Meta |
 | [opt-30b](https://arxiv.org/abs/2205.01068) | **82.09** +/- 0.47 | 30B |  Zhang et al., 2022 | Meta |
 | [opt-13b](https://arxiv.org/abs/2205.01068) | **81.94** +/- 0.46 | 13B |  Zhang et al., 2022 | Meta |
 | [gpt-neox-20b](https://arxiv.org/abs/2204.06745) | **81.50** +/- 0.47 | 20B |  Black et al., 2022 | EleutherAI |
 | [gpt-j-6b](https://github.com/kingoflolz/mesh-transformer-jax/#gpt-j-6B) |  **81.14** +/- 0.47 |  6B | Wang et al., 2021 | EleutherAI |
 | [pythia-12b](https://arxiv.org/abs/2304.01373) | **80.53** +/- 0.48 | 12B | Biderman et al., 2023 | EleutherAI|
 | [t5-v1-xxl](https://arxiv.org/abs/1910.10683) | **76.55** +/- 0.52 | 11B |  Raffel et al., 2019 | Google |
 | [bloom-7b1](https://arxiv.org/abs/2211.05100) | **76.16** +/- 0.51 |  7B |  Scao et al., 2022 | BigScience |
 | [gpt2-xl](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | **73.76** +/- 0.54 | 1.5B |  Radford et al., 2018 | OpenAI |
 | [bert-base](https://arxiv.org/abs/1810.04805) | **72.60** +/- 0.54 | 110M | Devlin et al., 2018 | Google | 
 | [m-bert-base](https://arxiv.org/abs/1810.04805) | **71.80** +/- 0.55 | 110M | Devlin et al., 2018 | Google | 
 | [stablelm-base-7b](https://github.com/Stability-AI/StableLM) | **68.85** +/- 0.55 | 7B | Stability et al., 2023 | Stability AI |
 | [mt5-xxl](https://arxiv.org/abs/2010.11934) | **61.58** +/- 0.59|  11B |  Xue et al., 2020 | Google |
 | [xlm-roberta-large](https://arxiv.org/abs/1911.02116) | **61.55** +/- 0.59 | 355M | Conneau et al., 2019 | Meta |
 | [mt5-xl](https://arxiv.org/abs/2010.11934) |  **59.96** +/- 0.59 | 3.7B |  Xue et al., 2020 | Google |
 | Random Baseline | 50   | &nbsp; | &nbsp; | &nbsp; |  
 
 **Table 2**: Insert caption.
 
 &nbsp;
 
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

* LLaMa weights were accessed with the approval of Meta AI and used in accordance with the License (see [link](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) for more details).  

[3] T-REx: A Large Scale Alignment of Natural Language with Knowledge Base Triples. ElSahar, Hady, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon S. Hare, Frédérique Laforest and Elena Paslaru Bontas Simperl. International Conference on Language Resources and Evaluation. [Link][trex] (2018).

[4] Mass Editing Memory in a Transformer. Meng, Kevin, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. arXiv preprint [arXiv:2210.07229][memit] (2022).


[bib]: https://github.com/daniel-furman/Polyglot-or-Not#bibliography
[hf_data]: https://huggingface.co/datasets/CalibraGPT/Fact-Completion
[cka]: https://arxiv.org/abs/2210.03329
[memit]: https://arxiv.org/abs/2210.07229
[mmlu]: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
[mmlu_paper]: https://arxiv.org/abs/2009.03300
[trex]: http://aclanthology.lst.uni-saarland.de/L18-1544.pdf
[cka_lightweight_demo]: https://github.com/daniel-furman/Polyglot-or-Not/blob/main/notebooks/fact_completion_notebooks/fact-completion-lightweight-demo.ipynb
[cka_run_main]: https://github.com/daniel-furman/Polyglot-or-Not/blob/main/notebooks/fact_completion_notebooks/fact-completion-full-benchmark.ipynb
