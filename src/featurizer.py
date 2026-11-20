import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors


def calculate_delaney_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    feats_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            feats_list.append({
                'MolLogP': np.nan, 'MolWt': np.nan,
                'NumRotatableBonds': np.nan, 'AromaticProportion': np.nan
            })
            continue

        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        heavy_atoms = mol.GetNumHeavyAtoms()
        aromatic_proportion = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0.0

        feats_list.append({
            'MolLogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticProportion': aromatic_proportion
        })
    return pd.DataFrame(feats_list)


def calculate_all_rdkit_2d_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    descriptors_data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptors_data.append([np.nan] * len(descriptor_names))
        else:
            descriptors_data.append(calculator.CalcDescriptors(mol))
            
    df_desc = pd.DataFrame(descriptors_data, columns=descriptor_names)
    df_desc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_desc


def generate_morgan_fingerprints(smiles_list: list[str], radius: int = 2, fp_size: int = 2048) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp_list = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_list.append(np.full(fp_size, np.nan))
        else:
            fp_list.append(generator.GetFingerprintAsNumPy(mol))
            
    col_names = [f'Morgan_{i}' for i in range(fp_size)]
    return pd.DataFrame(fp_list, columns=col_names)
