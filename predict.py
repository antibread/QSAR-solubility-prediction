import argparse
import os
import sys
import joblib
from rdkit import Chem

from src.featurizer import calculate_all_rdkit_2d_descriptors


def classify_solubility(log_s: float) -> str:
    if log_s > -1.0:
        return "Very soluble"
    elif log_s > -3.0:
        return "Soluble / Moderately soluble"
    elif log_s > -5.0:
        return "Poorly soluble"
    return "Practically insoluble"


def main():
    parser = argparse.ArgumentParser(
        description="Predict aqueous solubility (LogS) of a molecule from its SMILES string."
    )
    parser.add_argument(
        "smiles",
        type=str,
        help="SMILES representation of the molecule (e.g., 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "rf_rdkit2d_model.joblib"),
        help="Path to the trained model artifact (default: models/rf_rdkit2d_model.joblib)"
    )
    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        print(f"Error: Invalid SMILES string provided -> '{args.smiles}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(
            f"Error: Model file not found at '{args.model_path}'. Run 'python -m src.train' first.",
            file=sys.stderr
        )
        sys.exit(1)

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    medians = artifact["medians"]

    X_new = calculate_all_rdkit_2d_descriptors([args.smiles])
    X_new = X_new.fillna(medians)

    pred_logs = float(model.predict(X_new)[0])
    molar_solubility = 10 ** pred_logs
    category = classify_solubility(pred_logs)

    print("-" * 50)
    print(f"Input SMILES:        {args.smiles}")
    print(f"Predicted LogS:      {pred_logs:.3f} log(mol/L)")
    print(f"Molar Solubility:    {molar_solubility:.6f} mol/L")
    print(f"Solubility Class:    {category}")
    print("-" * 50)


if __name__ == "__main__":
    main()
