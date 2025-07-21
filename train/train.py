import torch
from torch_geometric.utils import negative_sampling

def train_node_classification(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.cross_entropy(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_node_classification(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).sum().item() / mask.sum().item()
    return acc

def get_link_labels(pos_edge_index, neg_edge_index):
    return torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])

def train_link_prediction(model, data, optimizer, pos_edge_index, neg_edge_index):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    scores = torch.cat([pos_score, neg_score])
    labels = get_link_labels(pos_edge_index, neg_edge_index).to(scores.device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def test_link_prediction(model, data, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        scores = torch.cat([pos_score, neg_score])
        labels = get_link_labels(pos_edge_index, neg_edge_index).to(scores.device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu(), scores.cpu())
    return auc 