from rdkit import Chem
from utils import embed_rdmol

sp3_smarts = [
    '[!R;CX4H2]', # CH2
    '[!R;CX4H]', # CH
    '[!R;CX4H0]', # C
    '[NX3H1;!R]', # NH
    '[NX3H0;!R]', # N
    '[OX2H0;!R]', # O
    '[#16X2H0;!R]', # S
]

sp2_smarts = [
    '[!R;CX3H1;!$([CX3H1](=O))]', # CH
    '[$([!R;#6X3H0]);!$([!R;#6X3H0]=[#8])]', # C
    '[#7X2H0;!R]', # N
    '[$([CX3H0](=[OX1]));!$([CX3](=[OX1])-[OX2]);!R]=O' # C=O
]

from rdkit.Chem import rdFreeSASA

def get_sasa(mol):
    if mol.GetNumConformers() == 0:
        mol = embed_rdmol(mol)
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    return sasa

def get_flexibility(mol):
    mol = Chem.AddHs(mol)
    n_sp3 = 0
    n_sp2 = 0
    for smarts in sp3_smarts:
        n_sp3 += len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
    for smarts in sp2_smarts:
        n_sp2 += len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))

    rings = Chem.GetSymmSSSR(mol)
    n_rings = len(rings)
    ring_atoms = [list(ring) for ring in rings]
    # If a ring has common atoms with another ring, adjust the number of rings
    for i in range(len(ring_atoms)):
        for j in range(i+1, len(ring_atoms)):
            if len(set(ring_atoms[i]) & set(ring_atoms[j])) > 0:
                n_rings -= 1
                break
    
    flexibility = n_sp3 + 0.5 * n_sp2 + 0.5 * n_rings - 1
    if flexibility < 0:
        flexibility = 0
    return flexibility