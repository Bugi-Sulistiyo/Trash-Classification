# Trash Image Classification

## Overview
This project aims to create a Convolutional Neural Network (CNN) model capable of classifying images of trash into five categories: cardboard, glass, metal, plastic, and trash.

## Requirements
You can find the necessary libraries and packages in the `requirements.txt` file.

## Dataset Information
The dataset used for this project can be found at the following URL: [TrashNet Dataset](https://huggingface.co/datasets/garythung/trashnet).

The zipped dataset is stored in the `dataset` directory of this project.

## Environment Setup
To set up your environment, you can use the following command to create a new conda environment (make sure you have conda installed):

```bash
pip install -r requirements.txt
```

## Training Process
To train the model, you can either run the script using GitHub Actions or manually execute the modeling script according to its alphabetical name.

The hyperparameters used for training can be found within the modeling script.

## Model Architecture
The model utilizes a Convolutional Neural Network (CNN) for classification tasks. Details regarding the architecture can be found in the modeling script.

## Data Augmentation
Data augmentation is handled in a separate script to reduce computational resources and streamline the training process.

## Checkpoint and Model Saving
Model checkpoints and saving processes are implemented within the training script. The saved models will be located in the designated output directory after training is complete in form of .h5 and .keras.

## Finding
The dataset utilized is relatively small, suggesting a strong need for data augmentation to enhance both its size and heterogeneity. Implementing augmentation techniques can significantly improve model performance by introducing variability in the training data. Additionally, the inclusion of the "trash" label may be reconsidered, as it does not align with the other categories and could introduce noise into the training process. Therefore, it may be beneficial to exclude this label to focus on the more relevant classes in the dataset.