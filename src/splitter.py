from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def get_scaffold_groups(smiles_list: list[str], include_chirality: bool = False) -> np.ndarray:
    groups = []
    for smi in smiles_list:
        scaffold = generate_scaffold(smi, include_chirality=include_chirality)
        if scaffold is None:
            groups.append("INVALID")
        elif scaffold == "":
            groups.append(f"acyclic_{smi}")
        else:
            groups.append(scaffold)
    return np.array(groups)


def scaffold_train_test_split(
    df: pd.DataFrame, smiles_col: str, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
):
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = sorted(
        scaffolds.values(),
        key=lambda x: (len(x), x[0]),
        reverse=True
    )

    train_cutoff = len(df) * (1 - test_size)
    train_idx, test_idx = [], []

    for scaffold_group in scaffold_sets:
        if len(train_idx) + len(scaffold_group) <= train_cutoff:
            train_idx.extend(scaffold_group)
        else:
            test_idx.extend(scaffold_group)

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
