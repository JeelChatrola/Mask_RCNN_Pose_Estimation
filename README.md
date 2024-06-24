# Mask R-CNN for Pose Estimation in Dynamic Environments

## Overview

This project enhances the Mask R-CNN framework with ResNet50 for human pose estimation, focusing on dynamic environments. The Real-time Pose Correction Feedback System is designed to assist users in maintaining proper form during physical exercises, starting with yoga, by providing real-time visual feedback.

## Key Features

- Utilizes Mask R-CNN with ResNet50 for accurate human pose estimation.
- Offers real-time feedback by comparing user's posture against ideal exercise forms.
- Targets yoga for initial analysis but is adaptable to various sports for posture improvement.

## Dataset

The project employs a subset of a larger yoga dataset, comprising 500 annotated images to ensure a diverse range of poses for comprehensive learning and analysis.

## Proposed Method

- **Model Architecture**: Integrates ResNet50 with Mask R-CNN for feature extraction and pose estimation.
- **Keypoint Detection**: Incorporates a dense layer in the network to predict x and y coordinates of key human joints.
- **Feedback System**: Analyzes captured poses against ideal ones to provide corrective feedback in real-time.
- **Training and Optimization**: Employs data augmentation, dropout, and fine-tuning strategies to enhance model performance and prevent overfitting.

## Results and Discussion

- The model demonstrates a stable training performance with notable accuracy in pose estimation.
- Real-world application potential in fitness, health monitoring, and sports training.
![Screenshot 2024-03-09 at 8 28 13 PM](https://github.com/shreyas-chigurupati07/Mask-RCNN-for-Dynamic-Environments/assets/84034817/819b5cae-e82a-4ee2-aa51-5e2dd69e841a)

## Conclusion and Future Work

The project illustrates the feasibility of using Mask R-CNN for pose estimation in dynamic settings, with potential expansion to various sports and improvements in model robustness and generalization.

## How to Use

1. Clone the repository and install the required dependencies.
2. Train the model using the provided dataset or your dataset following the annotation guidelines.
3. Deploy the model for real-time pose estimation and feedback.

## Dependencies

- Python
- TensorFlow
- Keras
- OpenCV
- Matplotlib (for plotting results)

