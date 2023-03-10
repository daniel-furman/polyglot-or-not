# Capstone Project

Materials for **"The Search for (Mis)Information in Large Language Models"** (MIMS Spring 2023 Capstone Project, UC Berkeley)

## Members

* Shreshta Bhat <bhat_shreshta@berkeley.edu>
* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Setup

* For running in Colab, see .ipynb files in ```./notebooks/```
    * Make sure to enable the GPU in your Colab session
    * See the "Open in Colab" button at the top of the notebooks
* For running locally, follow the steps below from the root dir
    * Running locally requires a Cuda GPU
    * A virtual env with python 3.9 is reccommended

```
pip install -r requirements.txt
cd src/cka_scripts
python run_cka.py configs.bert_v0
```
