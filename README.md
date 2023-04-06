# Measuring Multilingual Encyclopedic Knowledge Representation in Foundation Language Models

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/daniel-furman/Capstone/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This is the repository for [Measuring Multilingual Encyclopedic Knowledge Representation in Foundation Language Models](https://bit.ly/ischool-berkeley-capstone). It contains the following:

1. The [**code**][cka_run_main] for running the fact-completion benchmark with a compatible language model
2. The [**data**][hf_data] used for the fact-completion benchmark, which contains 20 languages
3. A [**demo**][cka_lightweight_demo] for fact-completion language models via contrastive knowledge assessment

The project's motivation is to assess how massively pre-trained language models handle matters of veracity and store factual associations across dozens of languages. In particular, we are focusing on foundation language models. Foundation language models are defined as models trained on broad data (generally using self-supervision at scale) that can be adapted to a wide range of downstream tasks.

## Test Leaderboard

To add a new model to the leaderboard, please reach out to us or submit a pull request.

| Model            | Authors      | English        |
|------------------|--------------|:--------------:|
| [`llama-30b`](https://arxiv.org/abs/2302.13971) | Touvron et al., 2023 | 89.40 +/- 0.38 |
| [`bloom-7b1`](https://arxiv.org/abs/2211.05100) | BigScience Workshop, 2022 | |
| [`xlm-roberta-large`](https://arxiv.org/abs/1911.02116) | Conneau et al., 2019 | |
| `Random guessing` | 50   | 

## Data Release

Cite calinet [[1][bib]], memit [[2][bib]], and t-rex [[3][bib]] papers.

## Compatible Language Model Families

| Model family | Release date | Model type | Organization |
|--------------|--------------|------------|--------------|
| `BERT`       | Oct 2018     | Masked LM  | Google       |
| `GPT2`       | Feb 2019     | Causal LM  | OpenAI       |
| `RoBERTa`    | Nov 2019     | Masked LM  | Meta AI      |
| `T5`         | Jul 2020     | Seq2Seq    | Google       |
| `GPT-J`      | Aug 2021     | Causal LM  | EleutherAI   |
| `GPT-NeoX`   | Feb 2022     | Causal LM  | EleutherAI   |
| `BLOOM`      | Mar 2022     | Causal LM  | Big Science  |
| `OPT`        | May 2022     | Causal LM  | Meta AI      |
| `Flan-T5`    | Dec 2022     | Seq2Seq    | Google       |
| `Pythia`     | Feb 2023     | Causal LM  | EleutherAI   |
| `LLaMa`      | Feb 2023     | Causal LM  | Meta AI      |
| `Flan-UL2`   | Mar 2023     | Seq2Seq    | Google       |

## Authors

* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>
* Shreshta Bhat <bhat_shreshta@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Citation

Please cite this repository as follows if you use its data or code:

```
@misc{Measuring_Multilingual_Encyclopedic_Knowledge_2023,
  author = {Daniel Furman and Tim Schott and Shreshta Bhat},
  title = {Measuring Multilingual Encyclopedic Knowledge Representation in Foundation Language Models},
  year = {2023}
  publisher = {GitHub},
  howpublished = {\url{https://github.com/daniel-furman/Capstone}},
}
```

## Bibliography 

[1] Dong, Qingxiu, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. "Calibrating Factual Knowledge in Pretrained Language Models". In Findings of the Association for Computational Linguistics: EMNLP 2022. [arXiv:2210.03329][cka] (2022).

[2] Meng, Kevin, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. "Mass Editing Memory in a Transformer." arXiv preprint [arXiv:2210.07229][memit] (2022).

[3] ElSahar, Hady, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon S. Hare, Frédérique Laforest and Elena Paslaru Bontas Simperl. “T-REx: A Large Scale Alignment of Natural Language with Knowledge Base Triples.” International Conference on Language Resources and Evaluation. [Link][trex] (2018).


[bib]: https://github.com/daniel-furman/Capstone#bibliography
[hf_data]: https://huggingface.co/datasets/CalibraGPT/Fact-Completion
[cka]: https://arxiv.org/abs/2210.03329
[memit]: https://arxiv.org/abs/2210.07229
[mmlu]: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
[mmlu_paper]: https://arxiv.org/abs/2009.03300
[trex]: http://aclanthology.lst.uni-saarland.de/L18-1544.pdf
[cka_lightweight_demo]: https://github.com/daniel-furman/Capstone/blob/main/notebooks/fact_completion_notebooks/fact-completion-lightweight-demo.ipynb
[cka_run_main]: https://github.com/daniel-furman/Capstone/blob/main/notebooks/fact_completion_notebooks/fact-completion-full-benchmark.ipynb
