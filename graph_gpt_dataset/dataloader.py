"""Utility dataloader for text-attributed graph datasets."""

from torch_geometric.loader import DataLoader


def create_dataloader(name: str, root: str = "data", batch_size: int = 32, shuffle: bool = True, **kwargs) -> DataLoader:
    """Instantiate a :class:`DataLoader` for a given TAGDataset.

    Parameters
    ----------
    name: str
        Name of the dataset (e.g., ``"cora"``, ``"ogbn-arxiv"``).
    root: str, optional
        Location to store and load the dataset.
    batch_size: int, optional
        Number of graphs per batch. Default is 32.
    shuffle: bool, optional
        Whether to shuffle the dataset each epoch. Default is ``True``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`torch_geometric.loader.DataLoader`.

    Returns
    -------
    DataLoader
        A data loader over ``TAGDataset``.
    """
    from . import TAGDataset

    dataset = TAGDataset(name=name, root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
