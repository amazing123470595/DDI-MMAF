import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer

# 直接复用你 dataloader 中的图像预处理
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
    
    # 模型解绑选项保留
    parser.add_argument("--unbind_weights", action="store_true", help="Unbind strict state_dict matching (sets strict=False).")
    parser.add_argument("--cpu_only", action="store_true", help="Unbind from GPU and force CPU inference.")
    
    return parser.parse_args()

def generate_composite_image(smiles1, smiles2, transform):
    """
    生成 224x448 的拼接图像，并应用 default_transform
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES sequence provided. Please check RDKit parsing.")
        
    img1 = Draw.MolToImage(mol1, size=(224, 224))
    img2 = Draw.MolToImage(mol2, size=(224, 224))
    
    # 创建空白画布进行水平拼接
    composite_img = Image.new('RGB', (448, 224))
    composite_img.paste(img1, (0, 0))
    composite_img.paste(img2, (224, 0))
    
    # 这里的 transform 就是你 dataloader 里的 default_transform
    img_tensor = transform(composite_img).unsqueeze(0) 
    return img_tensor

def main():
    args = parse_args()

    # 1. 硬件绑定/解绑
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tissue 检索
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

    # 3. 准备文本 (完全对齐训练时的模板和截断设置)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    text_input = f"{args.cell_line} is a {tissue} cancer cell line."
    
    # 将字典形式的 inputs 直接转移到 device 上
    inputs = tokenizer(
        [text_input], # 注意这里需要传列表，模拟 batch_size=1
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=64
    ).to(device)

    # 4. 准备图像
    image_tensor = generate_composite_image(args.smiles1, args.smiles2, default_transform).to(device)

    # 5. 模型加载与参数解绑控制
    model = MutiModelAF("dmis-lab/biobert-v1.1", config.num_classes).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
        
    strict_load = not args.unbind_weights
    model.load_state_dict(state_dict, strict=strict_load)
    model.eval()

    # 6. 推理与概率计算
    with torch.no_grad():
        # 修正：直接传入 images 和 inputs 字典，与训练代码保持一致
        outputs = model(image_tensor, inputs)
        
        # 因为训练代码用的是 CrossEntropyLoss，输出是 logits，所以用 softmax 计算概率
        probs = torch.softmax(outputs, dim=1)
        # 取类别 1 (协同) 的概率
        prob_pos = probs[0][1].item()
        
    # 7. 阈值判定与格式化输出
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