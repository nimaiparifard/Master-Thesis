"""Utilities for loading text-attributed graph datasets.

This module provides a :class:`TAGDataset` class that wraps several
popular citation and product graph datasets with textual node
attributes.  The implementation draws on dataset utilities from
`torch_geometric` and the Open Graph Benchmark (OGB).
"""

from pathlib import Path
from typing import List

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

from .dataloader import create_dataloader


class TAGDataset(InMemoryDataset):
    """Text-attributed graph dataset loader.

    Parameters
    ----------
    name: str
        Dataset name. Supported values are ``"cora"``, ``"pubmed"``,
        ``"citeseer"``, ``"ogbn-arxiv"`` and ``"ogbn-products"``.
    root: str, optional
        Root directory to store the dataset files.
    """

    def __init__(self, name: str, root: str = "data", **kwargs):
        self.name = name.lower()
        self.root_dir = Path(root) / self.name
        super().__init__(root=str(self.root_dir), **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # ------------------------------------------------------------------
    # InMemoryDataset requirements
    @property
    def raw_file_names(self) -> List[str]:  # pragma: no cover - handled internally
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):  # pragma: no cover - datasets handle their own download
        pass

    # ------------------------------------------------------------------
    def process(self):
        if self.name in {"cora", "pubmed", "citeseer"}:
            dataset = Planetoid(root=str(self.root_dir), name=self.name.capitalize())
            data = dataset[0]
            data.text = self._bow_to_text(data.x)
        elif self.name in {"ogbn-arxiv", "arxiv"}:
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(self.root_dir))
            data = dataset[0]
            data.y = data.y.squeeze()
            data.text = self._bow_to_text(data.x)
        elif self.name in {"ogbn-products", "products"}:
            dataset = PygNodePropPredDataset(name="ogbn-products", root=str(self.root_dir))
            data = dataset[0]
            data.y = data.y.squeeze()
            data.text = self._bow_to_text(data.x)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported dataset: {self.name}")

        torch.save(self.collate([data]), self.processed_paths[0])

    # ------------------------------------------------------------------
    @staticmethod
    def _bow_to_text(x: torch.Tensor) -> List[str]:
        """Convert bag-of-words style features to token strings.

        Each non-zero feature index is represented as ``word_{i}``.
        This provides simple textual content per node suitable for
        feeding into language models or text encoders.
        """

        texts: List[str] = []
        for row in x:
            idx = row.nonzero(as_tuple=False).view(-1).tolist()
            texts.append(" ".join(f"word_{i}" for i in idx))
        return texts

    # ------------------------------------------------------------------
    def get_text_embeddings(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        """Return sentence-transformer embeddings for node texts."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = self[0].text
        emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb.append(torch.tensor(model.encode(batch, show_progress_bar=False), dtype=torch.float))
        return torch.cat(emb, dim=0)

__all__ = ["TAGDataset", "create_dataloader"]

