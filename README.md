# Capstone Project

Materials for **"The Search for (Mis)Information in Large Language Models"** (MIMS Spring 2023 Capstone Project, UC Berkeley)

## Members

* Shreshta Bhat <bhat_shreshta@berkeley.edu>
* Daniel Furman <daniel_furman@berkeley.edu>
* Tim Schott <timschott@berkeley.edu>

## Advisor

* David Bamman <dbamman@berkeley.edu>

## Setup

* For running in Colab, see ipynbs in *./notebooks/*
    * Note that you should have the GPU enabled in Colab
* For running locally follow the steps below from the root dir
    * Note that a Cuda GPU is required for running locally 
    * Reccomended to run in a venv with python==3.9.16

```
pip install -r requirements.txt
cd src/cka_scripts
python run_cka.py configs.bert_v0
```
