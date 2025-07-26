# Master-Thesis

Utilities for text-attributed graph processing with large language models (LLMs).

## Project Overview

This project provides tools for processing, training, and evaluating text-attributed graph datasets using large language models. It supports node classification, link prediction, few-shot, and zero-shot learning tasks on datasets such as Cora, PubMed, and Arxiv.

## Setup

Install dependencies (requires PyTorch and Torch-Geometric):

```bash
pip install -r requirements.txt
```

## Loading Datasets

Run `load_datasets.py` to download and preprocess the Cora, PubMed, and Arxiv datasets:

```bash
python load_datasets.py
```

The script will create a `data/` directory containing the processed graph data with textual node attributes.

## Modules

- **models/**: Contains model definitions for text-attributed graph learning.
- **tagdatasets/**: Provides dataset classes for handling text-attributed graphs.
- **train/**: Training scripts and utilities for model training.
- **evaluations/**: Scripts for evaluating models on node classification, link prediction, few-shot, and zero-shot tasks.

## Training

To train a model, use:

```bash
python -m train.train
```

## Evaluation

To run evaluation tasks, use the scripts in the `evaluations/` directory. For example:

```bash
python -m evaluations.node_classifications
```

## Main Entry Point

You can also use `main.py` as a unified entry point for running training or evaluation workflows, depending on your configuration.
