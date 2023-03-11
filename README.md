# Capstone Project

Materials for **"The Search for (Mis)Information in Large Language Models"** (MIMS Spring 2023 Capstone Project, UC Berkeley)

## Members

* Shreshta Bhat <bhat_shreshta@berkeley.edu>
* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Models 

| Model family | Release date | Model type | Organization |
|--------------|--------------|------------|--------------|
| BERT         | Oct 2018     | Masked LM  | Google       |
| GPT2         | Feb 2019     | Causal LM  | OpenAI       |
| RoBERTa      | Nov 2019     | Masked LM  | Meta AI      |
| GPT-J        | Aug 2021     | Causal LM  | EleutherAI   |
| GPT-Neo      | Apr 2022     | Causal LM  | EleutherAI   |
| OPT          | May 2022     | Causal LM  | Meta AI      |
| Flan-t5      | Dec 2022     | Text2Text  | Google       |
| LLaMa        | Feb 2023     | Causal LM  | Meta AI      |
| Flan-ul2     | Mar 2023     | Text2Text  | Google       |

## Setup

* For running in Google Colab, see .ipynb files in ```./notebooks/```
    * Make sure to enable the GPU in your Colab session
    * See the "Open in Colab" button at the top of the notebooks
* For running locally, follow the steps below from the root dir
    * Running locally requires a Cuda GPU
    * A virtual env with python 3.9 is recommended

```
pip install -r requirements.txt
cd src/cka_scripts
python run_cka.py configs.bert_v0
```


