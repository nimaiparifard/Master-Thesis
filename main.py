from tagdatasets import TextGraphDataset
from models.model import GNNModel
from train.train import train_node_classification, test_node_classification
from evaluations.evaluation import zero_shot_node_classification, few_shot_node_classification
import torch
from tqdm import tqdm

if __name__ == "__main__":
    # Load dataset
    dataset = TextGraphDataset(name="cora")
    data = dataset[0]
    # Get SBERT embeddings
    data.x = dataset.get_text_embeddings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = GNNModel(data.x.size(1), 64, int(data.y.max()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Standard node classification
    for epoch in tqdm(range(1, 101), desc="Training Epochs"):  # tqdm progress bar
        loss = train_node_classification(model, data, optimizer, data.train_mask)
        if epoch % 10 == 0 or epoch == 1:
            acc = test_node_classification(model, data, data.test_mask)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    # Zero-shot node classification (example: mask out class 0 during training)
    unseen_class = 0
    unseen_class_mask = (data.y == unseen_class)
    acc_zero_shot = zero_shot_node_classification(model, data, unseen_class_mask)
    print(f"Zero-shot accuracy on class {unseen_class}: {acc_zero_shot:.4f}")

    # Few-shot node classification (example: only 5 nodes per class)
    k = 5
    few_shot_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    for c in range(int(data.y.max()) + 1):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            few_shot_mask[idx[:k]] = True
    acc_few_shot = few_shot_node_classification(model, data, few_shot_mask)
    print(f"Few-shot accuracy ({k} per class): {acc_few_shot:.4f}") 