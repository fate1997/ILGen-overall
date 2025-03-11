import os.path as osp
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles, negative_sampling


def build_il_graph(
    raw_dir: str, 
    edge_index_dict: Dict[str, torch.Tensor] = None
) -> HeteroData:
    r"""Build a heterogeneous graph for ionic liquids.
    
    Args:
        raw_dir (str): The directory of raw data.
        edge_index_dict (Dict[str, torch.Tensor], optional): The edge index 
            dictionary. Contains two keys: 's2t' and 't2s', is used to input 
            the edge index manually. Default: None.
    
    Returns:
        HeteroData: The heterogeneous graph for ionic liquids.
    """
    # Read cations, anions, and ILs
    anion_path = osp.join(raw_dir, 'IL-anions.csv')
    anions = pd.read_csv(anion_path)['smiles'].tolist()
    cation_path = osp.join(raw_dir, 'IL-cations.csv')
    cations = pd.read_csv(cation_path)['smiles'].tolist()
    il_path = osp.join(raw_dir, 'collected-ILs.csv')
    ils = pd.read_csv(il_path)['smiles'].tolist()

    # Build edge between anions and cations
    edges = []
    for il in ils:
        cation, anion = il.split('.')
        edges.append((cations.index(cation), anions.index(anion)))

    # Calculate ECFP4 features for cations and anions
    x_s = []
    x_t = []
    graph_s = []
    graph_t = []
    for cation in cations:
        mol = Chem.MolFromSmiles(cation)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        charge = Chem.GetFormalCharge(mol)
        fp = np.concatenate([fp, [charge]])
        graph_s.append(from_smiles(cation))
        x_s.append(fp)
    
    for anion in anions:
        mol = Chem.MolFromSmiles(anion)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        charge = Chem.GetFormalCharge(mol)
        fp = np.concatenate([fp, [charge]])
        graph_t.append(from_smiles(anion))
        x_t.append(fp)
    
    x_s = np.array(x_s)
    x_t = np.array(x_t)
    x_s = torch.tensor(x_s, dtype=torch.float)
    x_t = torch.tensor(x_t, dtype=torch.float)
    batch_s = next(iter(DataLoader(graph_s, batch_size=len(graph_s))))
    batch_t = next(iter(DataLoader(graph_t, batch_size=len(graph_t))))

    # Get edge index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    data = HeteroData()
    data['cation'].x = x_s
    data['anion'].x = x_t
    if edge_index_dict is None:
        data['cation', 'anion'].edge_index = edge_index
        data['anion', 'cation'].edge_index = edge_index[[1, 0]]
    else:
        data['cation', 'anion'].edge_index = edge_index_dict['s2t']
        data['anion', 'cation'].edge_index = edge_index_dict['t2s']
    data['cation'].smiles = cations
    data['anion'].smiles = anions
    data['cation'].graph = batch_s
    data['anion'].graph = batch_t
    return data


def split(
    data: HeteroData, 
    val_ratio: float, 
    test_ratio: float
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    r"""Split the data into training, validation, and test sets."""    
    # Clone the original data
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
    
    # Split the edge index from cation to anion
    num_edges = data['cation', 'anion'].edge_index.size(1)
    perm1 = torch.randperm(num_edges)
    val_mask = perm1 < int(val_ratio * num_edges)
    test_mask = (perm1 >= int(val_ratio * num_edges)) & \
                (perm1 < int((val_ratio + test_ratio) * num_edges))
    train_mask = perm1 >= int((val_ratio + test_ratio) * num_edges)
    
    val_data['cation', 'anion'].edge_index = data['cation', 'anion'].edge_index[:, val_mask]
    test_data['cation', 'anion'].edge_index = data['cation', 'anion'].edge_index[:, test_mask]
    train_data['cation', 'anion'].edge_index = data['cation', 'anion'].edge_index[:, train_mask]
    
    # Split the edge index from anion to cation
    perm2 = torch.randperm(num_edges)
    val_mask2 = perm2 < int(val_ratio * num_edges)
    test_mask2 = (perm2 >= int(val_ratio * num_edges)) & \
        (perm2 < int((val_ratio + test_ratio) * num_edges))
    train_mask2 = perm2 >= int((val_ratio + test_ratio) * num_edges)
    
    val_data['anion', 'cation'].edge_index = data['anion', 'cation'].edge_index[:, val_mask2]
    test_data['anion', 'cation'].edge_index = data['anion', 'cation'].edge_index[:, test_mask2]
    train_data['anion', 'cation'].edge_index = data['anion', 'cation'].edge_index[:, train_mask2]
    return train_data, val_data, test_data


def add_negative_samples(data: HeteroData) -> HeteroData:
    r"""Add negative samples for link prediction."""
    
    # Get the negative edge index
    neg_edge_index_s2t = negative_sampling(
        edge_index=data['cation', 'anion'].edge_index,
        num_nodes=(data['cation'].num_nodes, data['anion'].num_nodes),
        num_neg_samples=data['cation', 'anion'].edge_index.size(1))
    neg_edge_index_t2s = negative_sampling(
        edge_index=data['anion', 'cation'].edge_index,
        num_nodes=(data['anion'].num_nodes, data['cation'].num_nodes),
        num_neg_samples=data['anion', 'cation'].edge_index.size(1))
    
    # Concatenate the positive and negative edge index and edge labels
    edge_label_index_s2t = torch.cat(
        [data['cation', 'anion'].edge_index, neg_edge_index_s2t],
        dim=-1,
    )
    edge_label_index_t2s = torch.cat(
        [data['anion', 'cation'].edge_index, neg_edge_index_t2s],
        dim=-1,
    )
    edge_labels2t = torch.cat([
        data['cation', 'anion'].edge_index.new_ones(data['cation', 'anion'].edge_index.size(1)),
        data['cation', 'anion'].edge_index.new_zeros(neg_edge_index_s2t.size(1))
    ], dim=0)
    edge_labelt2s = torch.cat([
        data['anion', 'cation'].edge_index.new_ones(data['anion', 'cation'].edge_index.size(1)),
        data['anion', 'cation'].edge_index.new_zeros(neg_edge_index_t2s.size(1))
    ], dim=0)
    
    # Update the edge index and edge labels
    data['cation', 'anion'].edge_label_index = edge_label_index_s2t
    data['anion', 'cation'].edge_label_index = edge_label_index_t2s
    data['cation', 'anion'].edge_labels = edge_labels2t
    data['anion', 'cation'].edge_labels = edge_labelt2s
    return data