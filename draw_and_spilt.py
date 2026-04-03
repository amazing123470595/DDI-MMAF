import os
import pandas as pd
import hashlib
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# Global configuration
INPUT_FILES = ['breast.csv', 'colon.csv', 'lung.csv', 'melanoma.csv', 'ovarian.csv', 'prostate.csv']
IMAGE_DIR = "images"
SOURCE_DIR = "dataset"

def get_image_hash(s1, s2):
    # Generate unique MD5 hash for sorted SMILES pairs
    combined = "_".join(sorted([str(s1), str(s2)]))
    return hashlib.md5(combined.encode()).hexdigest() + ".png"

def get_scaffold(smiles):
    # Calculate Murcko Scaffold for a given SMILES string
    if pd.isna(smiles) or smiles == "":
        return "NA"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "NA"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except:
        return "NA"

def generate_unique_images(df):
    # Ensure the base directory exists
    # We no longer need output_dir as a parameter because the path is in the dataframe
    unique_pairs = df[['drug1', 'drug2', 'image_file']].drop_duplicates(subset=['image_file'])
    
    for _, row in unique_pairs.iterrows():
        img_path = row['image_file']
        
        # Create the directory if it's missing (e.g., "images/")
        dir_name = os.path.dirname(img_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        if not os.path.exists(img_path):
            m1, m2 = Chem.MolFromSmiles(row['drug1']), Chem.MolFromSmiles(row['drug2'])
            if m1 and m2:
                i1, i2 = Draw.MolToImage(m1, size=(224, 224)), Draw.MolToImage(m2, size=(224, 224))
                combined = Image.new('RGB', (448, 224))
                combined.paste(i1, (0, 0))
                combined.paste(i2, (224, 0))
                combined.save(img_path)

def save_split_results(train_df, test_df, prefix):
    # Assign sequential IDs and export dataframes to CSV
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)
    train.insert(0, 'id', range(1, len(train) + 1))
    test.insert(0, 'id', range(1, len(test) + 1))
    train.to_csv(f"{prefix}_train.csv", index=False)
    test.to_csv(f"{prefix}_test.csv", index=False)

def perform_random_stratified_split(df):
    # Partition dataset using random stratified sampling based on labels
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    cols_to_drop = ['scaffold1', 'scaffold2', 'scaffold_pair']
    save_split_results(
        train.drop(columns=cols_to_drop, errors='ignore'),
        test.drop(columns=cols_to_drop, errors='ignore'),
        "random"
    )

def perform_scaffold_group_split(df):
    # Partition dataset using GroupShuffleSplit based on molecular scaffolds
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['scaffold_pair']))
    save_split_results(df.iloc[train_idx], df.iloc[test_idx], "scaffold")

if __name__ == "__main__":
    data_list = []
    for file_name in INPUT_FILES:
        file_path = os.path.join(SOURCE_DIR, file_name)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            temp_df['source'] = os.path.splitext(file_name)[0]
            data_list.append(temp_df)

    if not data_list:
        print("No input files found.")
    else:
        master_df = pd.concat(data_list, ignore_index=True).dropna(subset=['drug1', 'drug2', 'label'])
        
        # 1. Correctly define image_file with the directory prefix
        master_df['image_file'] = master_df.apply(
            lambda r: os.path.join(IMAGE_DIR, get_image_hash(r['drug1'], r['drug2'])), 
            axis=1
        )

        print("Generating molecular images...")
        # 2. Pass the updated master_df to the generation function
        generate_unique_images(master_df)

        print("Calculating chemical scaffolds...")
        master_df['scaffold1'] = master_df['drug1'].apply(get_scaffold)
        master_df['scaffold2'] = master_df['drug2'].apply(get_scaffold)
        master_df['scaffold_pair'] = master_df.apply(
            lambda r: "_".join(sorted([r['scaffold1'], r['scaffold2']])), axis=1
        )

        print("Splitting datasets...")
        perform_random_stratified_split(master_df)
        perform_scaffold_group_split(master_df)

        print(f"Workflow completed. Total records: {len(master_df)}")