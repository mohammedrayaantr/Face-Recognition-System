# Face Recognition System

## Project Overview

This project demonstrates an educational face recognition system using Machine Learning and Computer Vision techniques.

The system detects faces using MTCNN, extracts numerical image features using EfficientNet Lite0, and classifies identities using a Support Vector Machine (SVM).

It also demonstrates predictions on new images and a basic confidence-based rejection mechanism.

## Project Objectives

- Detect faces in images.
- Extract numerical features from detected faces.
- Train a machine learning classifier to recognize known identities.
- Evaluate model performance.
- Test predictions on new images.
- Identify uncertain predictions using a confidence threshold.

## Technologies Used

- Python
- TensorFlow
- TensorFlow Hub
- EfficientNet Lite0
- MTCNN
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset

The project uses the **Labeled Faces in the Wild (LFW)** dataset.

The dataset was loaded using Scikit-learn's `fetch_lfw_people()` function with a minimum of 20 images per person.

Dataset details:

| Description | Value |
|---|---:|
| Images in filtered dataset | 3,023 |
| Number of identities | 62 |
| Successfully detected faces | 2,974 |
| Feature dimensions per face | 1,280 |

## Project Workflow

1. Load the LFW dataset.
2. Visualize sample images.
3. Detect faces using MTCNN.
4. Resize and preprocess detected faces.
5. Extract feature vectors using EfficientNet Lite0.
6. Encode identity labels.
7. Split the dataset into training and testing sets.
8. Train an SVM classifier.
9. Evaluate the classifier using accuracy, classification report, and confusion matrix.
10. Save and reload the trained model.
11. Test the model on new images.
12. Apply a basic confidence threshold to flag uncertain predictions.

## Model Performance

The SVM classifier achieved the following result on the held-out test dataset:

| Metric | Result |
|---|---:|
| Test Accuracy | 73.61% |
| Training Split | 80% |
| Testing Split | 20% |
| Test Samples | 595 |

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Face-Recognition-System
```

### 3. Install the required libraries

```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow tensorflow-hub mtcnn pillow certifi
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the project notebook

Open `Facial_Recognition_System.ipynb` and run the cells from beginning to end.

The LFW dataset is downloaded through Scikit-learn when the dataset-loading cell is executed. The external test image path in the notebook must be updated to match your own image location.

## Limitations

- This is an educational prototype, not a production-ready authentication system.
- The model classifies faces among the 62 identities included in the filtered dataset.
- EfficientNet Lite0 provides general image feature vectors rather than specialized face-recognition embeddings.
- The confidence threshold is experimental and has not been validated.
- The system does not implement liveness detection or anti-spoofing.
- Unknown-person verification is not reliably established.

## Future Improvements

- Experiment with face-recognition-specific embedding models.
- Improve classification performance.
- Calibrate the confidence threshold using validation data.
- Implement reliable unknown-person verification.
- Explore liveness detection and anti-spoofing techniques.

## Conclusion

This project provides practical experience with Computer Vision, deep learning-based feature extraction, traditional machine learning classification, model evaluation, and image-based prediction.

It demonstrates how different machine learning techniques can be combined to build an educational face recognition pipeline.
