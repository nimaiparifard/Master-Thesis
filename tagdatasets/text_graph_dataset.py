"""Text-attributed graph dataset utilities using PyTorch Geometric."""

from typing import List
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


class TextGraphDataset(InMemoryDataset):
    """Load and preprocess graph datasets with textual node attributes."""

    def __init__(self, name: str, root: str = "data", **kwargs):
        self.dataset_name = name.lower()
        self.root_dir = Path(root) / self.dataset_name
        super().__init__(root=str(self.root_dir), **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self):  # pragma: no cover - uses built-in datasets
        pass

    def process(self):
        if self.dataset_name in {"cora", "pubmed"}:
            dataset = Planetoid(root=str(self.root_dir), name=self.dataset_name.capitalize())
            data = dataset[0]
            data.text = self._bow_to_text(data.x)
        elif self.dataset_name == "arxiv":
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(self.root_dir))
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

    def get_text_embeddings(self, model_name='all-MiniLM-L6-v2', batch_size=64):
        """Return SBERT embeddings for node texts using the specified model."""
        from sentence_transformers import SentenceTransformer
        import torch
        model = SentenceTransformer(model_name)
        texts = self[0].text
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(torch.tensor(emb, dtype=torch.float))
        return torch.cat(embeddings, dim=0)
