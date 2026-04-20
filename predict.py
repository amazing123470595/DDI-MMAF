import argparse
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

from dataloader import default_transform
from model import MutiModelAF
import config

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:
    print("Warning: RDKit is not installed. SMILES to image generation will fail.")

def parse_args():
    parser = argparse.ArgumentParser(description="DDI-MMAF Synergy Prediction")
    parser.add_argument("--smiles1", type=str, required=True, help="SMILES sequence of drug 1")
    parser.add_argument("--smiles2", type=str, required=True, help="SMILES sequence of drug 2")
    parser.add_argument("--cell_line", type=str, required=True, help="Cancer cell line name (e.g., A2780)")
    parser.add_argument("--tissue", type=str, default=None, help="Tissue origin. If not provided, it will be looked up.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file")
    
    # Unbinding options
    parser.add_argument("--unbind_weights", action="store_true", help="Unbind strict state_dict matching (sets strict=False).")
    parser.add_argument("--cpu_only", action="store_true", help="Unbind from GPU and force CPU inference.")
    
    return parser.parse_args()

def generate_composite_image(smiles1, smiles2, transform):
    """Generate a 224x448 composite image from two SMILES sequences."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES sequence provided. Please check RDKit parsing.")
        
    img1 = Draw.MolToImage(mol1, size=(224, 224))
    img2 = Draw.MolToImage(mol2, size=(224, 224))
    
    composite_img = Image.new('RGB', (448, 224))
    composite_img.paste(img1, (0, 0))
    composite_img.paste(img2, (224, 0))
    
    img_tensor = transform(composite_img).unsqueeze(0) 
    return img_tensor

def main():
    args = parse_args()

    device = torch.device("cpu") if args.cpu_only else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tissue origin retrieval
    tissue = args.tissue
    if tissue is None:
        try:
            mapping_df = pd.read_csv("dataset/cell_tissue.csv")
            cell_to_text_dict = dict(zip(mapping_df['cell_line'], mapping_df['tissue']))
            tissue = cell_to_text_dict.get(args.cell_line)
            if tissue is None:
                raise ValueError(f"Tissue for '{args.cell_line}' not found.")
        except FileNotFoundError:
            raise FileNotFoundError("dataset/cell_tissue.csv not found.")

    # Text input preparation (matches training template)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    text_input = f"{args.cell_line} is a {tissue} cancer cell line."
    
    inputs = tokenizer(
        [text_input], 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=64
    ).to(device)

    image_tensor = generate_composite_image(args.smiles1, args.smiles2, default_transform).to(device)

    # Model initialization and weight loading
    model = MutiModelAF("dmis-lab/biobert-v1.1", config.num_classes).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
        
    strict_load = not args.unbind_weights
    model.load_state_dict(state_dict, strict=strict_load)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor, inputs)
        probs = torch.softmax(outputs, dim=1)
        prob_pos = probs[0][1].item()
        
    threshold = 0.5
    label = 1 if prob_pos >= threshold else 0

    print("-" * 50)
    print(f"SMILES 1 : {args.smiles1}")
    print(f"SMILES 2 : {args.smiles2}")
    print(f"Cell Line: {args.cell_line} ({tissue})")
    print(f"Prob(+)  : {prob_pos:.6f}")
    print(f"Label    : {label}  (threshold={threshold})")
    print("-" * 50)

if __name__ == "__main__":
    main()