import os
import random
from collections import defaultdict

def split_dataset(file_path, train_file, test_file, test_ratio=0.2):
    """
    Splits a dataset into train and test files with an equal distribution of each class.

    Args:
        file_path (str): Path to the input dataset file (format: <image_path> <class_id>).
        train_file (str): Path to the output train file.
        test_file (str): Path to the output test file.
        test_ratio (float): Ratio of the data to be used for testing (default: 0.2).
    """
    # Read the data
    class_data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                path, class_id = line.rsplit(' ', 1)
                class_data[class_id].append(line)

    train_data, test_data = [], []

    # Split the data for each class
    for class_id, items in class_data.items():
        random.shuffle(items)
        split_index = int(len(items) * test_ratio)

        # Ensure equal distribution by splitting the class samples proportionally
        test_samples = items[:split_index]
        train_samples = items[split_index:]
        train_data.extend(train_samples)
        test_data.extend(test_samples)

    # Write to the train and test files
    with open(train_file, 'w') as f_train:
        f_train.write('\n'.join(train_data))

    with open(test_file, 'w') as f_test:
        f_test.write('\n'.join(test_data))

    print(f"Data split complete:\nTrain file: {train_file}\nTest file: {test_file}")
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Example usage
file_path =r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg\data\combined.txt"  # Replace with the path to your dataset file
train_file = r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg\data\train.txt"
test_file = r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg\data\test.txt"

split_dataset(file_path, train_file, test_file, test_ratio=0.05)


# import os
# import random
# from collections import defaultdict

# def split_dataset(file_path, train_file, test_file, test_size):
#     """
#     Splits a dataset into train and test files with a fixed number of samples per class in the test set.

#     Args:
#         file_path (str): Path to the input dataset file (format: <image_path> <class_id>).
#         train_file (str): Path to the output train file.
#         test_file (str): Path to the output test file.
#         test_size (int): Number of samples to include in the test set for each class.
#     """
#     # Read the data
#     class_data = defaultdict(list)
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 path, class_id = line.rsplit(' ', 1)
#                 class_data[class_id].append(line)

#     train_data, test_data = [], []

#     # Split the data for each class
#     for class_id, items in class_data.items():
#         random.shuffle(items)
#         if len(items) < test_size:
#             print(f"Warning: Not enough samples in class {class_id}. Using all available ({len(items)}) as test samples.")
#             test_samples = items
#             train_samples = []  # No data left for training
#         else:
#             test_samples = items[:test_size]
#             train_samples = items[test_size:]

#         train_data.extend(train_samples)
#         test_data.extend(test_samples)

#     # Write to the train and test files
#     with open(train_file, 'w') as f_train:
#         f_train.write('\n'.join(train_data))

#     with open(test_file, 'w') as f_test:
#         f_test.write('\n'.join(test_data))

#     print(f"Data split complete:\nTrain file: {train_file}\nTest file: {test_file}")
#     print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# # Example usage
# file_path = r"T:\GAC\demographic_classifier\dataset-training\txt_files\Fairace_025_original_dataset_output.txt"  # Replace with the path to your dataset file
# train_file = r"T:\GAC\demographic_classifier\dataset-training\txt_files\train_Fairace_025_original_dataset.txt"
# test_file = r"T:\GAC\demographic_classifier\dataset-training\txt_files\test_Fairace_025_original_dataset.txt"
# test_size = 300  # Specify the number of samples to include in the test set for each class

# split_dataset(file_path, train_file, test_file, test_size)
