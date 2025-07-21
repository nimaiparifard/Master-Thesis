# Master-Thesis

Utilities for text-attributed graph processing with large language models (LLMs).

## Setup

Install dependencies (requires PyTorch and Torch-Geometric):

```bash
pip install -r requirements.txt
```

## Loading datasets

Run `load_datasets.py` to download and preprocess the Cora, PubMed and Arxiv datasets:

```bash
python load_datasets.py
```

The script will create a `data/` directory containing the processed graph data with textual node attributes.
