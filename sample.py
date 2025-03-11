import argparse
import pickle

import numpy as np
import pathlib
from moima.pipeline.vae_pipe import VAEPipe
from rdkit import Chem

from step3_post_filtering.train_logistic import featurize


REPO_DIR = pathlib.Path(__file__).resolve().parent


def main(
    vae_path: str,
    logistic_path: str,
    output_path: str,
    sample_size: int = 1000
):
    vae = VAEPipe.from_pretrained(vae_path, is_training=True)
    vae.logger.info(f'Sampling {sample_size} ILs...')
    sampled_smiles = vae.sample(sample_size)
    
    # Remove invalid smiles, duplicates, and not novel smiles
    vae.logger.info('Removing invalid smiles, duplicates, and not novel smiles...')
    sampled_smiles = [s for s in sampled_smiles if Chem.MolFromSmiles(s) is not None]
    sampled_smiles = list(set(sampled_smiles))
    train_smiles = vae.batch_flatten(vae.loader['train'], 
                                     register_items=['smiles'], 
                                     return_numpy=True, 
                                     register_output=False)['smiles']
    sampled_smiles = [s for s in sampled_smiles if s not in train_smiles]
    vae.logger.info(f'{len(sampled_smiles)} survived.')
    with open('sampled_smiles.txt', 'w') as f:
        for smiles in sampled_smiles:
            f.write(smiles + '\n')
    
    # Post-filtering
    vae.logger.info('Post-filtering...')
    X, mask = featurize(sampled_smiles)
    X = X[mask]
    sampled_smiles = np.array(sampled_smiles)[mask]
    
    with open(logistic_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    print(f'{(y_pred == 1).sum()}/{len(y_pred)} ILs passed the post-filtering.')
    sampled_smiles = sampled_smiles[y_pred == 1]
    
    # Check charge correctness
    vae.logger.info('Checking charge correctness...')
    finalized_smiles = []
    for smiles in sampled_smiles:
        if smiles.count('.') > 1:
            continue
        ion1, ion2 = smiles.split('.')
        ion1_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(ion1))
        ion2_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(ion2))
        if (ion1_charge + ion2_charge == 0) and (ion1_charge * ion2_charge < 0):
            finalized_smiles.append(smiles)
    
    # Write to file
    vae.logger.info(f'Writing {len(finalized_smiles)} ILs to {output_path}...')
    with open(output_path, 'w') as f:
        for smiles in finalized_smiles:
            f.write(smiles + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vae_path', 
        type=str,
        default=str(REPO_DIR / 'step2_vae_generation' / 'vae.pt')
    )
    parser.add_argument(
        '--logistic_path', 
        type=str,
        default=str(REPO_DIR / 'step3_post_filtering' / 'logistic.pkl')
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        default=str(REPO_DIR / 'sample.txt')
    )
    parser.add_argument(
        '--sample_size', 
        type=int,
        default=1000
    )
    main(**vars(parser.parse_args()))
