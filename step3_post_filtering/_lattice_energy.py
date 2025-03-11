from rdkit import Chem

from _constants import ATOM_VOLUME_CONTRIBUTIONS, GAS_CONSTANT, ureg
from _gas_entropy import GasEntropyModel

ENERGY_UNIT = 'kJ/mol'
DISTANCE_UNIT = 'nm'
ENTROPY_UNIT = 'kJ/(mol*K)'


def estimate_volume(mol: Chem.Mol) -> float:
    volume = 0
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        volume += ATOM_VOLUME_CONTRIBUTIONS[atomic_num] * 1e-3
    return volume


def estimate_solid_entropy(volume: float):
    volume = volume * ureg('nm^3')
    k = 1360 * ureg('J/(mol*K*nm^3)')
    c = 15 * ureg('J/(mol*K)')
    solid_entropy_298 = (k * volume + c) * 10 ** -3
    return solid_entropy_298.magnitude


def estimate_lattice_enthalpy(volume: float, temperature: float = 298):
    volume = volume * ureg('nm^3')
    temperature = temperature * ureg('K')
    alpha = 117.3 * ureg('kJ/mol')
    beta = 51.9 * ureg('kJ/mol')
    U_pot = 2 * (alpha / (volume / ureg('nm^3')) ** (1/3) + beta)
    H_latt = U_pot + 2 * GAS_CONSTANT * temperature
    return H_latt.magnitude


def estimate_lattice_energy(smiles: str, 
                            temperature: float = 298, 
                            return_entropy: bool = False):
    components = smiles.split('.')
    rdmols = [Chem.MolFromSmiles(comp) for comp in components]
    for rdmol in rdmols:
        if rdmol is None:
            return None
    
    # Total volume
    volumes = [estimate_volume(rdmol) for rdmol in rdmols]
    total_volume = sum(volumes)
    # Delta lattice enthalpy
    lattice_enthalpy = estimate_lattice_enthalpy(total_volume, temperature)
    # Delta lattice entropy
    solid_entropy = estimate_solid_entropy(total_volume)
    gas_entropies = GasEntropyModel().predict(rdmols, output_unit=ENTROPY_UNIT)
    gas_entropy = sum(gas_entropies)
    lattice_entropy = gas_entropy - solid_entropy
    # Delta lattice energy
    lattice_energy = lattice_enthalpy - lattice_entropy * temperature
    if return_entropy:
        return lattice_energy, lattice_entropy
    return lattice_energy


if __name__ == '__main__':
    solid_entroies = {
        0.229: 0.3264,
        0.287: 0.4053,
        0.388: 0.5427
    }
    for volume, entropy in solid_entroies.items():
        calculated_entropy = estimate_solid_entropy(volume)
        print(f'solid entropy: {calculated_entropy:.4f} (expected: {entropy:.4f})')
    
    import pandas as pd
    test_cases = pd.read_csv('test_cases.csv')
    for idx, row in test_cases.iterrows():
        volume = row.molecular_volume
        expected = row.delta_latt_H
        expected_energy = row.delta_latt_G
        result = estimate_lattice_enthalpy(volume)
        lattice_energy = estimate_lattice_energy(row.cation_smiles + '.' + row.anion_smiles)
        print(f'Case {idx + 1}: lattice enthalpy: {result:.3f} (expect {expected:.3f}), '
              f'lattice energy: {lattice_energy:.3f} (expect {expected_energy: .3f})')