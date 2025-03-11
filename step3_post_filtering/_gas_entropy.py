from typing import List, Literal, Union

import numpy as np
import pandas as pd
from lapy import TriaMesh
from rdkit import Chem, RDLogger
from skimage import measure
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from utils import embed_rdmol

RDLogger.DisableLog('rdApp.*')

INTERCEPT = 70.93073256874152
SCALE = 8.703000950558827

class GasEntropyModel:
    
    def __init__(self, 
                 intercept: float=INTERCEPT,
                 scale: float=SCALE,
                 vdw_threshold: float=0.1,
                 spacing: float=0.2,
                 padding: float=2,
                 sigma: float=0.1,
                 entropy_bins: int=64,
                 unit: Literal['J/(mol*K)', 'kJ/(mol*K)'] = 'J/(mol*K)'
                 ):
        self.intercept = intercept
        self.scale = scale
        self.vdw_threshold = vdw_threshold
        self.spacing = spacing
        self.padding = padding
        self.sigma = sigma
        self.entropy_bins = entropy_bins
        self.unit = unit
    
    def _gaussian(self, query_pos: np.array, mol: Chem.Mol):
        coords = mol.GetConformer().GetPositions()
        gaussian_sum = 0
        for atom in mol.GetAtoms():
            vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
            position = coords[atom.GetIdx()]
            dist = np.linalg.norm(position - query_pos, axis=1)
            gaussian = np.exp(-(dist - vdw_radius) / self.sigma)
            gaussian_sum += gaussian
        results = -self.sigma * np.log(gaussian_sum)
        return results
    
    def _vdw_surface(self, mol: Chem.Mol):
        coords = mol.GetConformer().GetPositions()
        min_coords = np.min(coords, axis=0)
        grid_min_coords = min_coords - self.padding
        max_coords = np.max(coords, axis=0)
        grid_max_coords = max_coords + self.padding
        
        x = np.arange(grid_min_coords[0], 
                      grid_max_coords[0] + self.spacing, 
                      self.spacing)
        y = np.arange(grid_min_coords[1], 
                      grid_max_coords[1] + self.spacing, 
                      self.spacing)
        z = np.arange(grid_min_coords[2], 
                      grid_max_coords[2] + self.spacing, 
                      self.spacing)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        pos = np.stack([xx, yy, zz]).transpose((1, 2, 3, 0)).reshape(-1, 3)
        values = self._gaussian(pos, mol).reshape(xx.shape)
        spacing = (self.spacing, self.spacing, self.spacing)
        verts, faces, _, _ = measure.marching_cubes(values, 
                                                    self.vdw_threshold, 
                                                    spacing=spacing)
        return verts, faces
    
    @staticmethod
    def _curvd_entropy(verts, faces):
        mesh = TriaMesh(verts, faces)
        _, _, k1, k2 = mesh.curvature_tria()
        S_face = 2 / np.pi * np.arctan((k1 + k2) / (k2 - k1))
        areas_face = np.array(mesh.tria_areas())
        
        prob_density, bins = np.histogram(S_face, bins=64, density=True)
        prob_per_bin = prob_density * np.diff(bins)
        bins[-1] += 1e-6
        shape_prob = prob_per_bin[np.digitize(S_face, bins) - 1]
        curvd_entropy = -np.sum(shape_prob * np.log2(shape_prob) * areas_face)
        return curvd_entropy
    
    def featurize(self, mol: Chem.Mol):
        mol = embed_rdmol(mol)
        mol = Chem.RemoveHs(mol)
        verts, faces = self._vdw_surface(mol)
        curvd_entropy = self._curvd_entropy(verts, faces)
        return curvd_entropy
    
    def __call__(self, 
                 mol: Chem.Mol, 
                 output_unit: Literal['J/(mol*K)', 'kJ/(mol*K)'] = 'J/(mol*K)'):
        curvd_entropy = self.featurize(mol)
        entropy = self.intercept + self.scale * curvd_entropy
        if self.unit == output_unit:
            return entropy
        elif self.unit == 'J/(mol*K)' and output_unit == 'kJ/(mol*K)':
            return entropy / 1000
        elif self.unit == 'kJ/(mol*K)' and output_unit == 'J/(mol*K)':
            return entropy * 1000
        else:
            raise ValueError(f"Cannot convert from {self.unit} to {output_unit}")
    
    def fit_df(self, df: pd.DataFrame, smiles_col: str, entropy_col: str):
        df['rdmols'] = df[smiles_col].apply(Chem.MolFromSmiles)
        df = df[df['rdmols'].notnull()]
        return self.fit(df['rdmols'], df[entropy_col])
    
    def fit(self, mols: List[Union[Chem.Mol, str]], entropies: List[float]):
        X = []
        for mol in tqdm(mols, desc='Calculating features'):
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            curvd_entropy = self.featurize(mol)
            X.append(curvd_entropy)
        X = np.array(X).reshape(-1, 1)
        y = np.array(entropies)
        reg = LinearRegression().fit(X, y)
        self.intercept = reg.intercept_
        self.scale = reg.coef_[0]
        return self
    
    def predict(self, 
                mols: List[Union[Chem.Mol, str]],
                output_unit: Literal['J/(mol*K)', 'kJ/(mol*K)'] = 'J/(mol*K)'):
        entropies = []
        for mol in mols:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            entropy = self(mol, output_unit)
            entropies.append(entropy)
        return entropies


if __name__ == '__main__':
    gas_entropies = {
        'CCN1C=C[N+](=C1)C': 0.3807,
        '[B-](F)(F)(F)F': 0.2700,
        'C(F)(F)(F)S(=O)(=O)[O-]': 0.3632,
        'CCCCN1C=C[N+](=C1)C': 0.4422,
        'F[P-](F)(F)(F)(F)F': 0.3130
    }
    for smiles, entropy in gas_entropies.items():
        mol = Chem.MolFromSmiles(smiles)
        estimator = GasEntropyModel()
        calculated_entropy = estimator(mol, output_unit='kJ/(mol*K)')
        print(f'{smiles} gas entropy: {calculated_entropy:.4f} (expected: {entropy:.4f})')