# Implementation of  Transition-based Model for Chinese Information Extraction

We provide sample sentences from ACE05 for illustration.
For full data, please refer to https://catalog.ldc.upenn.edu/ldc2006t06
Requirement: PyTorch (tested on 0.4), Python(v3)

## Setup

run train.py to start training



## Configuration

Configurations of the model and training are in config.py

Some important configurations:

* if\_gpu: whether to use GPU for training
* input\_dropout: whether to add dropout layer
* if\_pretrained: whether to using Glove pretrained embeddings


