# CalibraGPT: The Search for (Mis)Information in Large Language Models

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/daniel-furman/Capstone/blob/main/LICENSE) 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This is the repo for the UC Berkeley CalibraGPT project, which aims to probe and repair large amounts of factual knowledge committed to memory in language models. The repo contains the following:

1. The [**72.3k+ English data**][data] employed for the fact completion benchmark, 20+ more languages to come
2. A [**notebook demo**][notebook_cka_demo] for running lightweight contrastive knowledge assessment probing
3. The **code** for running the fact completion benchmark with a (Hugging Face compatible) language model

## Data Release

Cite calinet [[1][bib]] and memit [[2][bib]] data sources. 

## Selected benchmark results

**English** results:

| Model           | CalibraGPT (fact completion, this work)     | MMLU (multiple-choice answering)     |
|------------------|---------------------------------------------|------------------------------------------------|
| `llama-65b`    |    | 63.4%           |
| `llama-33b`    |    | 57.8%           |
| `flan-ul2`      |    | 52.2%           |
| `flan-t5-xll`   |    | 48.6%           |
| `llama-13b`    |    | 46.9%           |
| `flan-t5-xl`|    | 45.5%           |
| `flan-t5-large`|    | 40.5%           |
| `llama-7b`      | 85.68% (+/- 0.25%)    | 35.1%           |
| `flan-t5-base`  | 73.78% (+/- 0.32%)    | 33.7%           |
| `gpt-neox-20b`  |    | 33.6%           |
| `roberta-large` | 75.53% (+/- 0.31%)   | 27.9%           |
| `Random guessing` | 50%   | 25%           |

Add info on the CalibraGPT benchmark ... For reference, random guessing would score a 50%.

* The uncertainty estimates in the CalibraGPT results were calculated using bootstrap resampling, via 10,000 random samples with replacement and a 95% confidence level.  

Multi-task Language Understanding (MMLU) [[3][bib]] is a popular NLU benchmark comprising multiple-choice questions in 57 subjects (professional & academic). For reference, random guessing would score a 25%. 

* Details on MMLU scores above: The "few-shot, k=5" results are reported for auto-regressive models like `gpt-neox-20b` while the "CoT" results are reported for seq2seq models like `flan-t5-xll`. In contrast, for masked language models like `roberta-large`, the "fine-tuned" results are reported. 
* Performance of other models on MMLU: OpenAI's new `GPT-4` model scores a whopping 86.4% while the older `GPT-3.5` scores a 70.1%. These private-access models can't today be examined on the CalibraGPT benchmark, since the contrastive knowledge assessment method requires probing probabilities across multiple tokens in a model's vocabulary.
    * See the MMLU [leaderboard][mmlu] for more on the above score details and for the results of other models. 

**Multilingual** results (coming)

## Model families tested

| Model family | Release date | Model type | Organization |
|--------------|--------------|------------|--------------|
| `BERT`       | Oct 2018     | Masked LM  | Google       |
| `GPT2`       | Feb 2019     | Causal LM  | OpenAI       |
| `RoBERTa`    | Nov 2019     | Masked LM  | Meta AI      |
| `t5-v1_1`    | Jun 2021     | Seq2Seq    | Google       |
| `GPT-J`      | Aug 2021     | Causal LM  | EleutherAI   |
| `GPT-Neo`    | Apr 2022     | Causal LM  | EleutherAI   |
| `OPT`        | May 2022     | Causal LM  | Meta AI      |
| `Flan-t5`    | Dec 2022     | Seq2Seq    | Google       |
| `Pythia`     | Feb 2023     | Causal LM  | EleutherAI   |
| `LLaMa`      | Feb 2023     | Causal LM  | Meta AI      |
| `Flan-ul2`   | Mar 2023     | Seq2Seq    | Google       |

## Models Release

## Setup instructions

* For running a notebook in Google Colab, see .ipynb files in ```./notebooks/```
* For running locally, follow the steps below from the root dir

```
pip install -r requirements.txt
cd src/cka_scripts
python run_cka.py configs.tests.bert_v0
```

## Authors
All grad students below contributed equally.

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

## Bibliography 

1. Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li."Calibrating Factual Knowledge in Pretrained Language Models". In Findings of the Association for Computational Linguistics: EMNLP 2022. [arXiv:2210.03329][cka] (2022).
2. Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. "Mass Editing Memory in a Transformer." arXiv preprint [arXiv:2210.07229][memit] (2022).
3. Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. "Measuring massive multitask language understanding." arXiv preprint [arXiv:2009.03300][mmlu_paper] (2020).

[notebook_cka_demo]: https://colab.research.google.com/github/daniel-furman/Capstone/blob/main/notebooks/cka_run_main_demo.ipynb
[data]: https://github.com/daniel-furman/Capstone/tree/main/data/calibragpt_full_input_information.json
[cka]: https://arxiv.org/abs/2210.03329
[memit]: https://arxiv.org/abs/2210.07229
[mmlu]: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
[mmlu_paper]: https://arxiv.org/abs/2009.03300
[bib]: https://github.com/daniel-furman/Capstone#bibliography
