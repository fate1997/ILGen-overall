import os
import subprocess
from typing import Literal

from rdkit import Chem
from rdkit.Chem import AllChem


def embed_rdmol(mol):
    if mol.GetNumConformers() > 0:
        return mol
    mol_with_hs = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_hs)
    AllChem.UFFOptimizeMolecule(mol_with_hs)
    mol = Chem.RemoveHs(mol_with_hs)
    return mol

def get_cation_anion(smiles: str):
    cation, anion = smiles.split('.')
    cation_mol = Chem.MolFromSmiles(cation)
    anion_mol = Chem.MolFromSmiles(anion)
    cation_mol_charge = Chem.GetFormalCharge(cation_mol)
    anion_mol_charge = Chem.GetFormalCharge(anion_mol)
    if cation_mol_charge < 0 and anion_mol_charge > 0:
        cation_mol, anion_mol = anion_mol, cation_mol
    elif cation_mol_charge > 0 and anion_mol_charge < 0:
        pass
    else:
        raise ValueError(f'Cannot split {smiles} into cation and anion')
    return cation_mol, anion_mol

def embed_rdmol_(rdmol: Chem.Mol,
                workdir: str='./temp', 
                level: Literal['normal', 'tight', 'extreme'] = 'tight',) -> str:
    if rdmol.GetNumConformers() > 0:
        return rdmol
    
    original_dir = os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    
    mol_with_hs = Chem.AddHs(rdmol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    AllChem.EmbedMolecule(mol_with_hs, ps)
    if mol_with_hs.GetNumConformers() == 0:
        raise ValueError(f"Embedding failed for {Chem.MolToSmiles(rdmol)}")
    
    # Write molecule to xyz file
    xyz_path = "original.xyz"
    Chem.MolToXYZFile(mol_with_hs, xyz_path)
    
    # Run xtb geometry optimization and get optimized geometry
    charge = Chem.GetFormalCharge(rdmol)
    results = subprocess.run(["xtb", xyz_path, "--silent", "--opt", level, f"--charge", f"{charge}"],
                            check=True,
                            capture_output=True,
                            text=True,)
    with open('xtb.stdout', "w") as f:
        f.write(results.stdout)
    with open('xtb.stderr', "w") as f:
        f.write(results.stderr)
    
    # Set optimized geometry to mol
    new_mol = Chem.MolFromXYZFile("xtbopt.xyz")
    mol_with_hs.RemoveAllConformers()
    mol_with_hs.AddConformer(new_mol.GetConformer(0))
    
    # Add charges to the molecule
    with open('charges', 'r') as f:
        charges = f.readlines()
    for atom, charge in zip(mol_with_hs.GetAtoms(), charges):
        atom.SetDoubleProp('_GasteigerCharge', float(charge))
    
    os.chdir(original_dir)
    return mol_with_hs