import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision.models import  ResNet34_Weights
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import argparse
# ======================== Hyperparameters ========================

batch_size = 64
num_epochs = 50
learning_rate = 0.001
num_classes = 5  # Number of classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr_decay_step = 35
lr_decay_factor = 0.1

# Input data size
image_height = 112
image_width = 112
image_size = (image_height, image_width)


class_labels = {
    "asian": 0,
    "black": 1,
    "indian": 2,
    "white": 3,
    'latino_hispanic':4
}


label_to_race = {v: k for k, v in class_labels.items()}
# ======================== Data Preprocessing ========================
# Data augmentation and normalization for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# ======================== Custom Dataset Loader ========================
class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Initializes the dataset from a text file.

        Args:
            file_path (str): Path to the text file containing image paths and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.file_path = file_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Read the file and parse image paths and labels
        with open(file_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line:
                    img_path, label = line.rsplit(maxsplit=1)
                    #print(label)
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))  # Ensure labels are integers

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and label at the given index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is the transformed image and label is an integer.
        """
        image_path = self.image_paths[idx]
        try:
            image = datasets.folder.default_loader(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
        #image = datasets.folder.default_loader(image_path)  # Loads image from path
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Create training and validation datasets
def setup_dataset(train_txt, val_txt):
    train_dataset = CustomDataset(file_path=train_txt, transform=data_transforms['train'])
    val_dataset = CustomDataset(file_path=val_txt, transform=data_transforms['val'])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

# ======================== Model Setup ========================
# Load a pre-trained ResNet-18 model
model = models.resnet34(weights=None)
#model.fc = nn.Linear(model.fc.in_features, 18)  # 7 classes


# # print(state_dict)
# for k, p in state_dict.items():
#     print(k, p.shape)
# exit()
# Modify the fully connected layer to match our 4-class classification task
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, num_classes)  # 5 classes
)
#model.load_state_dict(torch.load(r"T:\GAC\demographic_classifier\dataset-training\BUPT_Balanced_Race_Classifier\best_model_epoch_18.pth"))

#freezing the extracting layer
for param in model.parameters():
    param.requires_grad = False
    
if isinstance(model.fc, nn.Sequential):
    in_features = model.fc[1].in_features  # Sequential Linear layer
else:
    in_features = model.fc.in_features     # Non-sequential Linear layer

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)

# Unfreeze the classifier for training
for param in model.fc.parameters():
    param.requires_grad = True
    
model = model.to(device)
# model.load_state_dict(torch.load(r"T:\GAC\demographic_classifier\BUPT_balanced_and_FaceArg_Combined\best_model_15.pth"))
# Apply Kaiming initialization
# def initialize_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

# model.apply(initialize_weights)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

optimizer_name = 'AdamW'
classifier = 'softmax'
backbone = 'resnet34'
model_name = 'resnet34'
def train_and_evaluate_with_race_logging(dataset, model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, backbone, learning_rate, batch_size, optimizer_name, classifier, num_classes, scheduler, checkpoint_dir, log_file):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Initialize a DataFrame to store training logs
    training_logs = pd.DataFrame(columns=[
        'CreatedAt', 'model_name', 'dataset', 'backbone', 'lr', 'batch_size', 'optimizer', 
        'classifier', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'race_accuracy', 
        'delete_this_row', 'Asian_accuracy', 'Black_accuracy', 'Indian_accuracy', 
        'White_accuracy', 'Latino_Hispanic_accuracy', 'epoch'
    ])
    
    best_val_accuracy = 0.0
    print("Starting training...")

    for epoch in range(num_epochs):
        # ======================== Training ========================
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        print(f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)

        # ======================== Evaluation ========================
        model.eval()
        running_val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1

        val_epoch_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)

        # Class-wise accuracies for Asian, Black, Indian, and White
        class_accuracies = {
            'Asian_accuracy': 100 * class_correct.get(0, 0) / class_total.get(0, 1),
            'Black_accuracy': 100 * class_correct.get(1, 0) / class_total.get(1, 1),
            'Indian_accuracy': 100 * class_correct.get(2, 0) / class_total.get(2, 1),
            'White_accuracy': 100 * class_correct.get(3, 0) / class_total.get(3, 1),
            'Latino_Hispanic_accuracy' :  100 * class_correct.get(4, 0) / class_total.get(4, 1)
        }

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        print("Class-wise accuracies:")
        for class_name, accuracy in class_accuracies.items():
            print(f" - {class_name}: {accuracy:.2f}%")

        current_lr = scheduler.get_last_lr()[0]
        # Save epoch metrics to DataFrame
        new_row = pd.DataFrame([{
            'CreatedAt': pd.Timestamp.now(),
            'model_name': model_name,
            'dataset': dataset,  # Replace with actual dataset name
            'backbone': backbone,
            'lr': current_lr,
            'batch_size': batch_size,
            'optimizer':  optimizer_name,
            'classifier': classifier,
            'train_loss': epoch_loss,
            'train_acc' : train_accuracy,
            'val_loss': val_epoch_loss,
            'val_acc': val_accuracy,
            'race_accuracy': val_accuracy,
            'delete_this_row': False,
            'Asian_accuracy': class_accuracies['Asian_accuracy'],
            'Black_accuracy': class_accuracies['Black_accuracy'],
            'Indian_accuracy': class_accuracies['Indian_accuracy'],
            'White_accuracy': class_accuracies['White_accuracy'],
            'Latino_Hispanic_accuracy': class_accuracies['Latino_Hispanic_accuracy'],
            'epoch': epoch + 1
        }])

        training_logs = pd.concat([training_logs, new_row], ignore_index=True)
        #checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        #torch.save(model.state_dict(), checkpoint_path)
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{epoch+1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved for epoch {epoch + 1}!")

        # Save metrics after each epoch
        logs_path = os.path.join(checkpoint_dir, log_file)
        training_logs.to_csv(logs_path, index=False)

    print("Training complete. Best validation accuracy:", best_val_accuracy)
    return training_logs


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with configurable paths and logging.")
    parser.add_argument('--dataset', type=str, default='FairFace', help='Dataset used for training')
    parser.add_argument('--checkpoint_dir', type=str, default=r'T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg_RFW_2_1', help='Directory to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='Fairface_UTK_RFW_downloaded_with_extraFrom_FaceArg_logs_1.csv', help='Name of the log file')
    parser.add_argument('--train_txt', type=str, default=r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg_RFW\data\train.txt", help='Path to the training text file')
    parser.add_argument('--val_txt', type=str, default=r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg_RFW\data\test.txt", help='Path to the validation text file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Dataset used for training: {args.dataset}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Log File: {args.log_file}")
    print(f"Training Data Path: {args.train_txt}")
    print(f"Validation Data Path: {args.val_txt}")
    
    train_loader, val_loader = setup_dataset(args.train_txt, args.val_txt)
    # Placeholder for actual training logic
    print("Starting training...")
    train_and_evaluate_with_race_logging(args.dataset, model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, backbone, learning_rate, batch_size, optimizer_name, classifier, num_classes, scheduler, args.checkpoint_dir, args.log_file)

    
if __name__ == "__main__":
    main()


# if __name__ == '__main__':
#     checkpoint_dir =
#     train_and_evaluate_with_race_logging(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, backbone, learning_rate, batch_size, optimizer_name, classifier, num_classes, scheduler, checkpoint_dir)
