# COVID-19 Detection from Chest X-ray Images Using CNN

This repository contains a Convolutional Neural Network (CNN) model designed to detect COVID-19 from chest X-ray images. The model is built using TensorFlow/Keras and performs binary classification (COVID vs. Non-COVID) on X-ray images. The code also includes an evaluation script to calculate accuracy and precision on a test dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview
The goal of this project is to identify COVID-19 infection based on chest X-ray images. The CNN model has been trained to differentiate between COVID-19 infected and non-COVID images, utilizing deep learning for effective feature extraction and classification.

## Dataset Structure
The dataset should be structured as follows:
- **`test_data/`** - Contains only non-COVID chest X-ray images (ground truth label = 0).
- **`test_data_2/`** - Contains only COVID-infected chest X-ray images (ground truth label = 1).

Both folders should contain images in `.jpg`, `.png`, or `.jpeg` formats.

## Model Architecture
The model uses a Convolutional Neural Network (CNN) for image classification. CNN layers extract spatial features from the X-ray images to distinguish between COVID-19 and non-COVID cases. The architecture includes several convolutional and pooling layers, followed by fully connected layers for classification.

## Prerequisites
- Python 3.7 or above
- TensorFlow 2.x
- NumPy
- OpenCV

You can install the required packages using:

```bash
pip install tensorflow numpy opencv-python
```
## Usage

Run the following command to start the prediction and evaluation script:

```bash
python testing.py
```
## Results
The output includes:

Total Images: The total number of test images processed.
True Positives: Correctly detected COVID-infected images.
False Positives: Non-COVID images incorrectly detected as COVID.
True Negatives: Correctly detected non-COVID images.
False Negatives: COVID-infected images incorrectly detected as non-COVID.
Accuracy and Precision of the model.

Example output:
```plaintext
Total number of images: 46
True Positives (COVID detected correctly): 17
False Positives (COVID incorrectly detected): 0
True Negatives (Non-COVID detected correctly): 20
False Negatives (Non-COVID incorrectly detected): 9

Accuracy: 80.43%
Precision for detecting COVID: 100.00%
```
## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code. See the [LICENSE](LICENSE) file for full license details.

MIT License
