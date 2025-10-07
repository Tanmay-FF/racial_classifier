import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse

# ======================== ARGUMENT PARSER ========================
def parse_arguments():
    parser = argparse.ArgumentParser(description="5-Class Inference Script")
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing images')
    parser.add_argument('--model_path', type=str, default=r"models\racial_classifier_5classes.pth", help='Path to trained model')
    parser.add_argument('--output_csv', type=str, default=r"inference_results\predictions.csv", help='Output CSV file for predictions')
    return parser.parse_args()

args = parse_arguments()

# ======================== CONFIG ==================================
num_classes = 5
class_labels = {
    "asian": 0,
    "black": 1,
    "indian": 2,
    "white": 3,
    "latino_hispanic": 4
}
label_map = {v: k for k, v in class_labels.items()}

# ======================== TRANSFORMATIONS =============================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================== LOAD MODEL ==================================
model = models.resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(model.fc.in_features, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# ======================== GATHER IMAGE PATHS ========================
image_paths = []
for root, _, files in os.walk(args.root_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

# ======================== INFERENCE ==================================
results = []
with torch.no_grad():
    for img_path in tqdm(image_paths, desc="Running inference"):
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

            results.append({
                "img_path": img_path,
                "predicted_label": label_map[predicted.item()]
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# ======================== SAVE RESULTS ==================================
df = pd.DataFrame(results)
df.to_csv(args.output_csv, index=False)
print(f"Predictions saved to: {args.output_csv}")
