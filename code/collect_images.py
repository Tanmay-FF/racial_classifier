import os
import random

# Define the root folder path
root_folder = r"T:\GAC\demographic_classifier\RFW"

class_labels = {
    "asian": 0,
    "black": 1,
    "indian": 2,
    "white": 3,
    'latino_hispanic' : 4
}

# # Define the number of images to take from each folder
images_to_take = {
    "asian": 28477,
    "black": 28477,
    "indian": 28477,
    "white": 28477,
    'latino_hispanic' : 28477
}
# Initialize a list to hold image paths and their labels
data = []

# Traverse through each class folder as defined in images_to_take
for class_name, num_images in images_to_take.items():
    if class_name in class_labels:  # Ensure the class exists in the mapping
        class_folder = os.path.join(root_folder, class_name)
        
        # Check if the class folder exists
        if os.path.exists(class_folder):
            image_files = []
            
            # Walk through the directory and its subdirectories
            for dirpath, _, files in os.walk(class_folder):
                for filename in files:
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if needed
                        image_files.append(os.path.join(dirpath, filename))
            
            # Randomly select the specified number of images or all available if less than required
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            # Add selected images to data list with their corresponding label
            for img_path in selected_images:
                data.append((img_path, class_labels[class_name]))
        else:
            print(f"Warning: Folder '{class_folder}' does not exist.")

# Function to write data to a text file
def write_to_file(data, filename):
    with open(filename, 'w') as f:
        for img_path, label in data:
            f.write(f"{img_path} {label}\n")


# Write data to a text file
save_path = os.path.join(r"dataset-training/Fairface_UTK_downloaded/data",'RFW_all.txt')
write_to_file(data,save_path )

print("Output file created successfully with selected images.")