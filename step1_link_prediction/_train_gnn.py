import logging

import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from _data import add_negative_samples, HeteroData


def train_step(
    model: nn.Module, 
    data: HeteroData, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: str = 'cuda'
) -> torch.Tensor:
    """Train the model for one step."""
    model.to(device)
    model.train()
    optimizer.zero_grad()
    data.to(device)
    z_s, z_t = model.encode(data)

    # We perform a new round of negative sampling for every training epoch:
    data = add_negative_samples(data)
    data.to(device)
    edge_label_index_s2t = data['cation', 'anion'].edge_label_index
    edge_label_index_t2s = data['anion', 'cation'].edge_label_index
    edge_labels2t = data['cation', 'anion'].edge_labels
    edge_labelt2s = data['anion', 'cation'].edge_labels
    edge_label = torch.cat([edge_labels2t, edge_labelt2s], dim=0).float()

    out_s2t = model.decode(z_s, z_t, edge_label_index_s2t).view(-1)
    out_t2s = model.decode(z_t, z_s, edge_label_index_t2s).view(-1)
    out = torch.cat([out_s2t, out_t2s], dim=0)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(
    model: nn.Module, 
    data: HeteroData, 
    device: str = 'cuda'
) -> tuple[float, float]:
    """Test the model."""
    model.eval()
    model.to(device)
    data.to(device)
    
    z_s, z_t = model.encode(data)
    out_s2t = model.decode(z_s, z_t, data['cation', 'anion'].edge_label_index)
    out_t2s = model.decode(z_t, z_s, data['anion', 'cation'].edge_label_index)
    out_s2t = out_s2t.view(-1)
    out_t2s = out_t2s.view(-1)
    
    out = torch.cat([out_s2t, out_t2s], dim=0)
    groud_truth_s2t = data['cation', 'anion'].edge_labels
    ground_truth_t2s = data['anion', 'cation'].edge_labels
    ground_truth = torch.cat([groud_truth_s2t, ground_truth_t2s], dim=0).float()
    auc = roc_auc_score(ground_truth.cpu().numpy(), out.cpu().numpy())
    acc = ((out > 0).float() == ground_truth).sum().item() / ground_truth.size(0)
    return auc, acc

def train(
    model: nn.Module, 
    train_data: HeteroData, 
    val_data: HeteroData, 
    test_data: HeteroData, 
    logger: logging.Logger = None
):
    """Train the model."""
    if logger is None:
        logger = logging.getLogger(__name__)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train_step(model, train_data, optimizer, criterion)
        val_auc, val_acc = test(model, val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Auc: {val_auc:.4f}, '
                    f'Val Acc: {val_acc:.4f}')

    test_auc, test_acc = test(model, test_data)
    logger.info(f'Test Auc: {test_auc:.4f}, Test Acc: {test_acc:.4f}')