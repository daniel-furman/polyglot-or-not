# CalibraGPT: The Search for (Mis)Information in Large Language Models

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/daniel-furman/Capstone/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This is the repo for the UC Berkeley CalibraGPT project, which aims to probe and repair large amounts of factual knowledge committed to memory in large language models. The repo contains the following:

1. A [**notebook demo**][notebook_cka_demo] for contrastive knowledge assessment probing
2. The [**72.3k+ data**][data] employed for our CalibraGPT factual associations benchmark
3. Supporting **web app demos** (to come)
4. Supporting **models** (to come)

## Data Release

(To come)

## Models Release

(To come)

## Benchmark

(Results to come)

| Model family | Release date | Model type | Organization |
|--------------|--------------|------------|--------------|
| `BERT`       | Oct 2018     | Masked LM  | Google       |
| `GPT2`       | Feb 2019     | Causal LM  | OpenAI       |
| `RoBERTa`    | Nov 2019     | Masked LM  | Meta AI      |
| `t5-v1_1`    | Jun 2021     | Seq-2-Seq  | Google       |
| `GPT-J`      | Aug 2021     | Causal LM  | EleutherAI   |
| `GPT-Neo`    | Apr 2022     | Causal LM  | EleutherAI   |
| `OPT`        | May 2022     | Causal LM  | Meta AI      |
| `Flan-t5`    | Dec 2022     | Seq2Seq    | Google       |
| `Pythia`     | Feb 2023     | Causal LM  | EleutherAI   |
| `LLaMa`      | Feb 2023     | Causal LM  | Meta AI      |
| `Flan-ul2`   | Mar 2023     | Seq2Seq    | Google       |

## Setup instructions

* For running a notebook in Google Colab, see .ipynb files in ```./notebooks/```
* For running locally, follow the steps below from the root dir

```
pip install -r requirements.txt
cd src/cka_scripts
python run_cka.py configs.tests.bert_v0
```

## Authors
All grad students below contributed equally. The order is determined alphabetically by last name. 

* Shreshta Bhat <bhat_shreshta@berkeley.edu>
* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{calibragpt,
  author = {Shreshta Bhat and Daniel Furman and Tim Schott},
  title = {CalibraGPT: The Search for (Mis)Information in Large Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/daniel-furman/Capstone}},
}
```

Naturally, you should also cite the original Contrastive Knowledge Assessment paper [[1][cka]] and Mass-Editing Memory in a Transformer paper [[2][memit]]. 

1. Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li."Calibrating Factual Knowledge in Pretrained Language Models". In Findings of the Association for Computational Linguistics: EMNLP 2022. [arXiv:2210.03329][cka] (2022).
2. Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. "Mass Editing Memory in a Transformer." arXiv preprint [arXiv:2210.07229][memit] (2022).

[notebook_cka_demo]: https://colab.research.google.com/github/daniel-furman/Capstone/blob/main/notebooks/cka_run_main_demo.ipynb
[data]: https://github.com/daniel-furman/Capstone/tree/main/data/calibragpt_full_input_information.json
[cka]: https://arxiv.org/abs/2210.03329
[memit]: https://arxiv.org/abs/2210.07229
