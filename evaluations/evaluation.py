import torch
import numpy as np

def zero_shot_node_classification(model, data, unseen_class_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[unseen_class_mask].argmax(dim=1)
        acc = (pred == data.y[unseen_class_mask]).sum().item() / unseen_class_mask.sum().item()
    return acc

def few_shot_node_classification(model, data, few_shot_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[few_shot_mask].argmax(dim=1)
        acc = (pred == data.y[few_shot_mask]).sum().item() / few_shot_mask.sum().item()
    return acc

def zero_shot_link_prediction(model, data, unseen_pos_edge_index, unseen_neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pos_score = (z[unseen_pos_edge_index[0]] * z[unseen_pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[unseen_neg_edge_index[0]] * z[unseen_neg_edge_index[1]]).sum(dim=1)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones(unseen_pos_edge_index.size(1)),
            torch.zeros(unseen_neg_edge_index.size(1))
        ]).to(scores.device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu(), scores.cpu())
    return auc

def few_shot_link_prediction(model, data, few_shot_pos_edge_index, few_shot_neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        pos_score = (z[few_shot_pos_edge_index[0]] * z[few_shot_pos_edge_index[1]]).sum(dim=1)
        neg_score = (z[few_shot_neg_edge_index[0]] * z[few_shot_neg_edge_index[1]]).sum(dim=1)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones(few_shot_pos_edge_index.size(1)),
            torch.zeros(few_shot_neg_edge_index.size(1))
        ]).to(scores.device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.cpu(), scores.cpu())
    return auc 