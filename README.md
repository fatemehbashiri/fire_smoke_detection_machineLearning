# fire_smoke_detection_machineLearning
using GNB model to detect fire and smoke in video

# Smoke Detection in Videos

## Overview

This Python script is designed for smoke detection in videos using computer vision techniques and machine learning. It processes video frames, applies background subtraction for motion detection, performs color thresholding, and utilizes a pre-trained machine learning model for smoke classification.

## Features

- **Motion Detection:** Utilizes a background subtraction technique to identify areas of motion in video frames.
- **Color Thresholding:** Converts frames to different color spaces and applies thresholds to identify relevant parts related to smoke.
- **Object Detection and Tracking:** Detects contours in the binary mask and filters out small contours to track potential smoke regions.
- **Feature Extraction:** Extracts features from detected regions and uses a pre-trained Gaussian Naive Bayes (GNB) model for smoke classification.
- **Visualization:** Draws rectangles around detected smoke regions on the original frames.
- **Output:** Writes processed frames to an output video file.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PyWavelets
- scikit-image
- scikit-learn
- Matplotlib

Install the required packages using:

```bash
pip install -r requirements.txt
