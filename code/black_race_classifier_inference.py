import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import argparse

# ======================== ARGUMENT PARSER ========================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for Black/Other classifier")
    parser.add_argument('--root_dir', type=str, required=True, help="Root folder containing image subdirectories")
    parser.add_argument('--model_path', type=str, default=r"models\black_classifier.pth", help="Path to the trained model (.pth)")
    parser.add_argument('--output_csv', type=str, default=r"inference_results\black_classifier_predictions.csv", help="CSV file to save predictions")
    return parser.parse_args()

args = parse_arguments()

# ======================== CONFIG ========================
root_dir = args.root_dir
model_path = args.model_path
output_csv = args.output_csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
black_label = {"black": 0, "other": 1}
label_to_race = {v: k for k, v in black_label.items()}
num_classes = len(black_label)

# ======================== TRANSFORM ========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ======================== LOAD MODEL ========================
model = models.resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================== LOAD DATA ========================
image_paths = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(subdir, file))

# ======================== INFERENCE ========================
results = []

with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Running inference"):
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

            pred_label = label_to_race[predicted.item()]
            results.append({
                "img_path": img_path,
                "predicted_label": pred_label
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# ======================== SAVE CSV ========================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
