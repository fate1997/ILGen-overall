import argparse
import logging
import os
import pathlib

import numpy as np
import pandas as pd
from moima.utils._util import get_logger
from sage_gnn import SAGE

from _data import add_negative_samples, build_il_graph, split
from _train_gnn import train

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent


def run_single_round(
    raw_dir: str, 
    edge_index_dict: dict = None,
    logger: logging.Logger = None
):
    # Build data
    data = build_il_graph(raw_dir, edge_index_dict)
    train_data, val_data, test_data = split(data, 0.05, 0.1)
    val_data = add_negative_samples(val_data)
    test_data = add_negative_samples(test_data)
    
    # Train model
    model = SAGE(in_channels=2049, hidden_channels=512, out_channels=256)
    train(model, train_data, val_data, test_data, logger)
    
    # Predict edges
    data.to('cpu')
    model.to('cpu')
    z_s, z_t = model.encode(data)
    new_edge_index_s2t = model.decode_all(z_s, z_t)
    new_edge_index_t2s = model.decode_all(z_t, z_s)
    edge_index_dict = {'s2t': new_edge_index_s2t, 't2s': new_edge_index_t2s}
    return edge_index_dict, data


def run_multiple_rounds(
    raw_dir: str,
    num_rounds: int,
    output_dir: str,
    logger: logging.Logger = None
):
    edge_index_dict = None
    for i in range(num_rounds):
        logger.info('#' * 50)
        logger.info(f'Round {i + 1}'.center(50, '#'))
        logger.info('#' * 50)
        edge_index_dict, data = run_single_round(raw_dir, edge_index_dict, logger)
        logger.info(f'Number of new edges: {int(edge_index_dict["s2t"].size(1))}')
        
        edge_index = edge_index_dict['s2t'].cpu().detach().numpy()
        src, dst = edge_index
        cation_smiles = np.array(data['cation'].smiles)[src]
        anion_smiles = np.array(data['anion'].smiles)[dst]
        il_smiles = np.array([f'{c}.{a}' for c, a in zip(cation_smiles, anion_smiles)])
        df = pd.DataFrame({'smiles': il_smiles})
        df.to_csv(os.path.join(str(output_dir), f'expanded_ILs_{i + 1}.csv'), index=False)
    
    edge_index = edge_index_dict['s2t'].cpu().detach().numpy()
    src, dst = edge_index
    cation_smiles = np.array(data['cation'].smiles)[src]
    anion_smiles = np.array(data['anion'].smiles)[dst]
    il_smiles = np.array([f'{c}.{a}' for c, a in zip(cation_smiles, anion_smiles)])
    
    df = pd.DataFrame({'smiles': il_smiles})
    df.to_csv(os.path.join(str(output_dir), 'expanded-ILs.csv'), index=False)


if __name__ == '__main__':
    raw_dir = REPO_DIR / 'database'
    output_dir = pathlib.Path(__file__).resolve().parent / 'output'
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', type=int, default=7)
    args = parser.parse_args()
    
    logger = get_logger('run_multiple_rounds.log', output_dir)
    run_multiple_rounds(raw_dir, args.num_rounds, output_dir, logger)