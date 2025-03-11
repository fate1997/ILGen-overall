import pathlib
import pickle
from typing import List, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from utils import get_cation_anion
from _flexibility import get_flexibility, get_sasa
from _lattice_energy import estimate_lattice_energy

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent

MP_PATH = REPO_DIR / 'database/IL-melting-point.csv'

def featurize(
    smiles_list: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    energies = []
    flexibilities = []
    sasas = []
    mask = []
    for smiles in tqdm(smiles_list, desc='Featurizing'):
        try:
            cation, anion = get_cation_anion(smiles)
            flexibility = (get_flexibility(cation), get_flexibility(anion))
            sasa = (get_sasa(cation), get_sasa(anion))
            energy = estimate_lattice_energy(smiles)
            mask.append(True)
        except Exception as e:
            flexibility = (np.nan, np.nan)
            sasa = (np.nan, np.nan)
            energy = np.nan
            mask.append(False)
        energies.append(energy)
        flexibilities.append(flexibility)
        sasas.append(sasa)
    
    flexibilities = np.array(flexibilities)
    sasas = np.array(sasas)
    energies = np.array(energies).reshape(-1, 1)
    return np.hstack([flexibilities, sasas, energies]), np.array(mask)


def train(
    mp_path: str,
    output_path: str
) -> None:
    mp = pd.read_csv(mp_path)
    X, mask = featurize(mp['smiles'])
    X = X[mask]
    y = mp['Melting point (K)'][mask] < 373 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    model_path = pathlib.Path(__file__).resolve().parent / 'logistic.pkl'
    train(MP_PATH, model_path)
