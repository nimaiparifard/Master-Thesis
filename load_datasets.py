"""Example script showing how to load text-attributed graph datasets."""

import logging
import os

from tagdatasets.text_graph_dataset import TextGraphDataset


if __name__ == "__main__":
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ]
    )

    best_val_acc = 0.0
    best_model_path = "best_model.pt"

    for name in ["cora", "pubmed", "arxiv"]:
        print(f"Loading {name} dataset")
        dataset = TextGraphDataset(name=name)
        data = dataset[0]
        print(data)
        print(f"Sample text attribute for node 0: {data.text[0][:100]}\n")
        # Example: Get SBERT embeddings for node texts
        embeddings = dataset.get_text_embeddings()
        print(f"SBERT embedding shape: {embeddings.shape}")
        print(f"First embedding (truncated): {embeddings[0][:5]}")

