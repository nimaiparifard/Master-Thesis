"""Example script showing how to load text-attributed graph datasets."""

from datasets.text_graph_dataset import TextGraphDataset


if __name__ == "__main__":
    for name in ["cora", "pubmed", "arxiv"]:
        print(f"Loading {name} dataset")
        dataset = TextGraphDataset(name=name)
        data = dataset[0]
        print(data)
        print(f"Sample text attribute for node 0: {data.text[0][:100]}\n")

