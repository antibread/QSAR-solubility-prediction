# 🧪 QSAR Modeling: Aqueous Solubility Prediction (ESOL)

A regression-based **QSAR (Quantitative Structure-Activity Relationship)** modeling project designed to predict the aqueous solubility (log S) of organic compounds directly from their chemical structures (SMILES).

## 📌 Overview
Aqueous solubility is a critical physicochemical parameter in drug discovery (ADME profiling) that directly impacts a compound's bioavailability. This project implements an end-to-end cheminformatics pipeline: from molecular descriptor generation and data preprocessing to machine learning model validation and virtual screening of novel molecules.

## 📊 Dataset & Molecular Descriptors
The model is trained on the benchmark **Delaney (ESOL)** dataset, which contains **1,144 organic compounds** with experimentally measured aqueous solubility values (log S).

Using **RDKit**, four classic Delaney physicochemical descriptors were calculated to featurize the SMILES strings:
1. **MolLogP** — Octanol-water partition coefficient (lipophilicity).
2. **MolWt** — Molecular weight.
3. **NumRotatableBonds** — Number of rotatable bonds (molecular flexibility).
4. **AromaticProportion** — Ratio of aromatic atoms to the total number of heavy atoms.

## ⚙️ Tech Stack
* **Language:** Python 3.10+
* **Cheminformatics:** `RDKit`
* **Machine Learning:** `Scikit-learn` (Random Forest Regressor)
* **Data Analysis & Visualization:** `Pandas`, `NumPy`, `Matplotlib`

## 📈 Results & Validation
A **Random Forest Regressor** was trained using an 80/20 train-test split (`random_state=42`). Model robustness and generalization were verified using **5-Fold Cross-Validation**:

| Evaluation Stage | R^2 Score | RMSE |
| :--- | :---: | :---: |
| **5-Fold Cross-Validation** | `0.86 ± 0.02` | `0.78` |
| **Hold-out Test Set** | **`0.89`** | **`0.69`** |

### Model Interpretability (Feature Importance)
Feature importance analysis confirmed the underlying physicochemical principles: lipophilicity (**MolLogP**) is the primary driver of solubility predictions (>80% relative importance), followed by molecular weight (**MolWt**).


## 🔬 Virtual Screening of Novel Compounds
The trained model was tested on well-known pharmaceutical compounds excluded from the training set:

| Compound | Predicted log S | Molar Solubility (mol/L) | Solubility Classification |
| :--- | :---: | :---: | :--- |
| **Caffeine** | `-0.98` | `0.10471` | Very soluble |
| **Paracetamol** | `-1.25` | `0.05623` | Soluble |
| **Aspirin** | `-1.82` | `0.01514` | Moderately soluble |
| **Ibuprofen** | `-3.65` | `0.00022` | Poorly soluble |
| **Cholesterol** | `-6.42` | `0.0000004` | Practically insoluble |

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd qsar-solubility-prediction
