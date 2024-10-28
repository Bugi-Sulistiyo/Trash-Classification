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