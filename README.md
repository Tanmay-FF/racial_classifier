# Race Classifier: Facial Image-Based Race Prediction 🏆

## Objective 🎯

This project is focused on the development of a highly accurate **race classifier** that predicts race based on facial images. The classifier is trained using a large dataset of labeled facial images from various racial groups and is intended to provide useful insights into demographic prediction from facial data.

The **primary goal** is to create a race classifier capable of:
1. **Training**: Utilize a comprehensive dataset to train the model.
2. **Inference**: Perform race classification on new, unseen facial images.
3. **Comparative Analysis**: Assess the performance of this custom model against an existing production model.

## Features 🌟

- **Accurate Race Prediction**: Predict race labels based on facial images (Asian, Black, Indian, White, Latino/Hispanic).
- **Customizable**: Easily adaptable to different datasets and configurations.
- **Inference**: Use the trained model for real-time race classification on new facial images.
- **ONNX Support**:  The trained model can be easily converted to ONNX for easy deployment and faster inference.

---

## Project Setup 🚀

Follow the instructions below to get your environment set up and run the project. 

### Prerequisites 🛠️

Before starting, ensure you have the following installed:
- `Python 3.6+`
- `pandas==2.2.3`
- `scikit_learn==1.4.0`
- `torch==2.2.1`
- `torchvision==0.17.1`
- `tqdm==4.66.1`
- `opencv` (optional for custom image processing)
- `pillow` (for image manipulation)

You can install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Dataset 📚

This model is trained on the **FairFace dataset**, **UTK-dataset** and **Face-Arg dataset**, which consists of diverse facial images categorized by race (Asian, Black, Indian, White, and Latino/Hispanic). We have also scraped a ton of data from the internet as well. Each class approximately contains `50,000` images to train upon.

You’ll need to download the dataset and prepare it in a specific format.

The dataset should be structured with the following:
- **Training images**: Path to images and labels stored in `train.txt`.
- **Validation images**: Path to images and labels stored in `val.txt`.

Each line in the `.txt` files should contain an image path followed by the corresponding label.

---

## Training Instructions 🏋️‍♂️

Once your environment is set up, you can start training the model. Follow these steps:

### Step 1: Prepare Your Data

Ensure that you have the dataset prepared in the following format.

Example structure:
```
/root_dir
    /asian
        /identity_1
          image1.jpg
          image2.jpg 1
        /identity_2
          image1.jpg
          image2.jpg 1
        ...
    /black
        /identity_1
          image1.jpg
          image2.jpg 1
        /identity_2
          image1.jpg
          image2.jpg 1
        ...
    ...
```

### Step 2: Set Hyperparameters

You can adjust the training parameters in the script to suit your needs:

```python
batch_size = 64
num_epochs = 50
learning_rate = 0.001
```

### Step 3: Start Training

To begin training, run the training script with the following command:

```bash
python train.py --dataset <dataset_name> --checkpoint_dir <checkpoint_save_path> --log_file <log_file_name> --train_txt <path_to_train_txt> --val_txt '<path_to_val_txt> 
```

This will start the training process, and the model will save the best performing checkpoint during training. The training progress will also be logged into a CSV file, which includes loss, accuracy, and race-specific accuracy metrics.

### Monitoring Training 🧑‍🏫

During training, the system will log:
- **Training loss**
- **Training accuracy**
- **Validation loss**
- **Validation accuracy**
- **Race-wise accuracy** for each race category (Asian, Black, Indian, White, Latino/Hispanic)

You will see an output in the console like this after each epoch:

```
Epoch 1/50 | Train Loss: 0.5754, Train Accuracy: 83.21% | Val Loss: 0.4234, Val Accuracy: 87.75%
Class-wise accuracies:
 - Asian_accuracy: 88.12%
 - Black_accuracy: 90.15%
 - Indian_accuracy: 85.36%
 - White_accuracy: 88.97%
 - Latino_Hispanic_accuracy: 87.14%
```

---

## Performance Evaluation 📊

After training, the model’s performance is evaluated based on:
- **Overall accuracy**: Accuracy across all race categories.
- **Class-wise accuracy**: Accuracy for each race group (Asian, Black, Indian, White, Latino/Hispanic).

These metrics are saved in a CSV log file after every epoch.

### Comparative Analysis 🆚

We also compare the performance of the trained model with an existing **production model**. You can benchmark the model by loading the production model and running it on the same validation dataset for comparison.

[Link to the comparison](https://facefirst.atlassian.net/wiki/spaces/MLResearch/pages/2415001607/ML+Experiment+-+Race+Classifier)

---

## Model Evaluation Example:

```bash
# Use a pre-trained model for evaluation:
python evaluation.py --root_dir "path_to_your_images" --model_path "path_to_your_model.pth" --output_csv "output_results.csv" --output_html "output_report.html"
```

This script will compare the performance of your model on the validation dataset.

---

## Future Goals 🚀

This project is continuously evolving, with plans to expand its capabilities beyond race classification. Future developments include:

1. **Integration of Gender Classification** 🏳️‍🌈  
   - Adding the ability to predict gender (Male, Female, Non-binary) alongside race.  
   - Improve fairness and debiasing techniques for better representation across all demographics.

2. **Age Group Classification** 🎂  
   - Implementing an age classification model to categorize individuals into different age groups (e.g., Child, Teen, Adult, Senior).  
   - Refining dataset augmentation techniques to improve generalization across age groups.
