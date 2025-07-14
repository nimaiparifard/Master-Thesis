"""Text-attributed graph dataset utilities using PyTorch Geometric."""

from typing import List
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid, OGBNArxiv


class TextGraphDataset(InMemoryDataset):
    """Load and preprocess graph datasets with textual node attributes."""

    def __init__(self, name: str, root: str = "data", **kwargs):
        self.dataset_name = name.lower()
        self.root_dir = Path(root) / self.dataset_name
        super().__init__(root=str(self.root_dir), **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):  # pragma: no cover - uses built-in datasets
        pass

    def process(self):
        if self.dataset_name in {"cora", "pubmed"}:
            dataset = Planetoid(root=str(self.root_dir), name=self.dataset_name.capitalize())
            data = dataset[0]
            data.text = self._bow_to_text(data.x)
        elif self.dataset_name == "arxiv":
            dataset = OGBNArxiv(root=str(self.root_dir))
            data = dataset[0]
            data.text = self._bow_to_text(data.x)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        torch.save(self.collate([data]), self.processed_paths[0])

    @staticmethod
    def _bow_to_text(x: torch.Tensor) -> List[str]:
        """Convert bag-of-words vectors to space separated word tokens."""
        tokens = []
        for row in x:
            idx = row.nonzero(as_tuple=False).view(-1).tolist()
            tokens.append(" ".join(f"word_{i}" for i in idx))
        return tokens
