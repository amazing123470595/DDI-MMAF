# DDI-MMAF: Multi-Modal Affine Fusion of Visual and Semantic Representations for Anticancer Drug Synergy Prediction

This repository contains the implementation of **DDI-MMAF**, a deep learning framework designed to predict drug-drug interactions (DDI) using multi-modal data. The pipeline includes molecule image generation, dataset partitioning (Random and Scaffold splits), and model training.

## 1. Installation

First, create a virtual environment and install the necessary dependencies:

```bash
# Create the environment
conda create -n ddi_mmaf python=3.9 -y

# Activate the environment
conda activate ddi_mmaf

# Install required packages
pip install -r requirements.txt
```

## 2. Data Preprocessing & Splitting

Run the preprocessing script to generate molecular images and partition the dataset. This script supports both **Random Split** and **Scaffold Split** (to evaluate model generalization on unseen molecular structures).

```bash
python draw_and_spilt.py
```

**After execution, the following files will be generated in the root directory:**
* `random_train.csv` / `random_test.csv`: Training and testing sets via random shuffling.
* `scaffold_train.csv` / `scaffold_test.csv`: Training and testing sets based on Bemis-Murcko scaffolds.

## 3. Model Training

Before starting the training process, ensure your configuration parameters are set correctly.

### 3.1 Configuration
Modify the parameters in your configuration file (or the relevant script section) to point to your desired dataset. For example, to train using the **Random Split** data:

```python
# Example configuration setup
train_csv = 'random_train.csv'
test_csv = 'random_test.csv'
```

### 3.2 Start Training
Once the configuration is ready, execute the main script to begin training:

```bash
python main.py
```

## 4. Using DDI-MMAF for drug synergy prediction

### Single combination prediction

We provide a simple command-line interface to predict the synergistic effect of a specific drug combination on a given cancer cell line. For example, to predict the synergy of two given SMILES strings on the A2780 cell line using a pre-trained model:

```bash
python predict.py \
    --smiles1 "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1" \
    --smiles2 "CCc1c2c(nc3ccc(O)cc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC" \
    --cell_line "A2780" \
    --model_path "checkpoints/best_model.pth"
```

Output example:

```text
--------------------------------------------------
SMILES 1 : C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1
SMILES 2 : CCc1c2c(nc3ccc(O)cc13)-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC
Cell Line: A2780 (ovarian)
Prob(+)  : 0.980479
Label    : 1  (threshold=0.5)
--------------------------------------------------
```
