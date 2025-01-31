# import os
# import torch
# import torch.nn as nn
# import pandas as pd
# from torchvision import transforms, models
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from sklearn.metrics import classification_report

# num_classes = 5

# facearg_class_labels = {
#     "asian": 0,
#     "black": 1,
#     "indian": 2,
#     "white": 3,
#     "latino_hispanic" : 4
# }

# # Reverse mapping to decode predicted labels
# label_map = {v: k for k, v in facearg_class_labels.items()}

# # Define the dataset class
# class ImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []

#         # Traverse through the root_dir and get image paths and labels
#         for label in os.listdir(root_dir):
#             label_dir = os.path.join(root_dir, label)
#             if os.path.isdir(label_dir):
#                 for subdir in os.listdir(label_dir):
#                     subdir_path = os.path.join(label_dir, subdir)
#                     if os.path.isdir(subdir_path):
#                         for img_name in os.listdir(subdir_path):
#                             img_path = os.path.join(subdir_path, img_name)
#                             if img_path.endswith(('jpg', 'jpeg', 'png')):
#                                 self.image_paths.append(img_path)
#                                 self.labels.append(facearg_class_labels[label])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label, img_path

# # Load your model (replace with your own model)
# model = models.resnet34(weights=None)
# model.fc = nn.Sequential(
#     nn.Dropout(0.25),
#     nn.Linear(model.fc.in_features, num_classes)
# )
# model.to('cuda:0')
# model.load_state_dict(torch.load(r"T:\GAC\demographic_classifier\dataset-training\FaceArg_downloadLatino_FairFaceLatino_Combined_Round_6_with_allRace_from_UTK\best_model_2.pth"))
# model.eval()  # Set the model to evaluation mode

# # Define transformations (if necessary, e.g., resizing, normalization)
# transform = transforms.Compose([
#     transforms.Resize((112, 112)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Set the root directory path
# root_dir = r"T:\\GAC\\demographic_classifier\\RFW"  # Change this to your actual root directory

# # Create the dataset and dataloader
# dataset = ImageDataset(root_dir=root_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # Prepare for recording predictions
# results = []
# all_true_labels = []
# all_predicted_labels = []

# # Prediction loop
# with torch.no_grad():
#     for inputs, original_labels, img_paths in tqdm(dataloader):
#         # Move to GPU if needed
#         inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs

#         # Run the model prediction
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)

#         # Convert predicted label to the corresponding string (map your label index to label name)
#         predicted_label = label_map[predicted.item()]
#         original_label_str = label_map[original_labels.item()]

#         # Append the image path, original label, and predicted label to results
#         results.append({
#             'img_path': img_paths[0],
#             'original_label': original_label_str,
#             'predicted_label': predicted_label
#         })

#         # Store true and predicted labels for classification report
#         all_true_labels.append(original_labels.item())
#         all_predicted_labels.append(predicted.item())

# # Save the results to a CSV file
# df = pd.DataFrame(results)
# df.to_csv("BUPT_and_FaceArg_combined_model_RFW_prediction_results.csv", index=False)

# # Generate HTML report
# html_content = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Classification Results</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             background-color: #f4f4f9;
#             margin: 0;
#             padding: 0;
#             color: #333;
#         }
#         h1 {
#             text-align: center;
#             color: #2c3e50;
#             margin-top: 30px;
#         }
#         table {
#             width: 80%;
#             margin: 20px auto;
#             border-collapse: collapse;
#             background-color: #ffffff;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#         }
#         th, td {
#             padding: 12px;
#             text-align: center;
#             border: 1px solid #ddd;
#         }
#         th {
#             background-color: #3498db;
#             color: white;
#         }
#         td {
#             background-color: #ecf0f1;
#         }
#         td img {
#             max-width: 150px;
#             height: auto;
#             border-radius: 8px;
#         }
#         tr:hover {
#             background-color: #f9f9f9;
#         }
#     </style>
# </head>
# <body>
#     <h1>Model Classification Results</h1>
#     <table>
#         <tr>
#             <th>Image</th>
#             <th>True Label</th>
#             <th>Predicted Label</th>
#         </tr>
# """

# # Populate the table with the results
# for result in results:
#     html_content += f"""
#     <tr>
#         <td><img src='{result['img_path']}' alt='Image'></td>
#         <td>{result['original_label']}</td>
#         <td>{result['predicted_label']}</td>
#     </tr>
# """

# # Close the table and HTML content
# html_content += """
#     </table>
# </body>
# </html>
# """

# # Save the HTML content to a file
# with open("BUPT_model_RFW_classification_results.html", "w") as f:
#     f.write(html_content)

# print("HTML report generated and saved successfully.")

import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
import argparse

# Set up argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Face recognition classification script.")
    parser.add_argument('--root_dir', type=str, default=r"T:\GAC\racial_classifier\sample_images", help='Root directory for image dataset')
    parser.add_argument('--model_path', type=str, default=r"T:\GAC\demographic_classifier\dataset-training\FaceArg_downloadLatino_FairFaceLatino_Combined_Round_6_with_allRace_from_UTK\best_model_2.pth", help='Path to the trained model')
    parser.add_argument('--output_csv', type=str, default="sample_result.csv", help='Output CSV file to store results')
    parser.add_argument('--output_html', type=str, default="sample_result.html", help='Output HTML file to store results')

    return parser.parse_args()

# Parse the arguments
args = parse_arguments()

num_classes = 5

facearg_class_labels = {
    "asian": 0,
    "black": 1,
    "indian": 2,
    "white": 3,
    "latino_hispanic" : 4
}

# Reverse mapping to decode predicted labels
label_map = {v: k for k, v in facearg_class_labels.items()}

# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse through the root_dir and get image paths and labels
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for subdir in os.listdir(label_dir):
                    subdir_path = os.path.join(label_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for img_name in os.listdir(subdir_path):
                            img_path = os.path.join(subdir_path, img_name)
                            if img_path.endswith(('jpg', 'jpeg', 'png')):
                                self.image_paths.append(img_path)
                                self.labels.append(facearg_class_labels[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# Load your model (replace with your own model)
model = models.resnet34(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(model.fc.in_features, num_classes)
)
model.to('cuda:0')
model.load_state_dict(torch.load(args.model_path))  # Use the model path from arguments
model.eval()  # Set the model to evaluation mode

# Define transformations (if necessary, e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the dataset and dataloader
dataset = ImageDataset(root_dir=args.root_dir, transform=transform)  # Use root_dir from arguments
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Prepare for recording predictions
results = []
all_true_labels = []
all_predicted_labels = []

# Prediction loop
with torch.no_grad():
    for inputs, original_labels, img_paths in tqdm(dataloader):
        # Move to GPU if needed
        inputs = inputs.to('cuda') if torch.cuda.is_available() else inputs

        # Run the model prediction
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Convert predicted label to the corresponding string (map your label index to label name)
        predicted_label = label_map[predicted.item()]
        original_label_str = label_map[original_labels.item()]

        # Append the image path, original label, and predicted label to results
        results.append({
            'img_path': img_paths[0],
            'original_label': original_label_str,
            'predicted_label': predicted_label
        })

        # Store true and predicted labels for classification report
        all_true_labels.append(original_labels.item())
        all_predicted_labels.append(predicted.item())

# Save the results to a CSV file
df = pd.DataFrame(results)
df.to_csv(args.output_csv, index=False)  # Use the output CSV file path from arguments

# Generate HTML report
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            color: #333;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-top: 30px;
        }
        table {
            width: 80%;
            margin: 20px auto;
            border-collapse: collapse;
            background-color: #ffffff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        th, td {
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        td {
            background-color: #ecf0f1;
        }
        td img {
            max-width: 150px;
            height: auto;
            border-radius: 8px;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>Model Classification Results</h1>
    <table>
        <tr>
            <th>Image</th>
            <th>True Label</th>
            <th>Predicted Label</th>
        </tr>
"""

# Populate the table with the results
for result in results:
    html_content += f"""
    <tr>
        <td><img src='{result['img_path']}' alt='Image'></td>
        <td>{result['original_label']}</td>
        <td>{result['predicted_label']}</td>
    </tr>
"""

# Close the table and HTML content
html_content += """
    </table>
</body>
</html>
"""

# Save the HTML content to a file
with open(args.output_html, "w") as f:
    f.write(html_content)

print("HTML report generated and saved successfully.")
