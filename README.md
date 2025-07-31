SVD-Based Handwritten Digit Classification

This project implements a classification algorithm for handwritten digits using **Singular Value Decomposition (SVD)**. It was developed as part of a miniproject on pattern recognition and scientific computing.

🔍 Project Overview

The goal is to classify grayscale images of handwritten digits (0–9) using linear algebra techniques, specifically the **SVD** of image matrices. Each image is represented as a vector of 784 values (flattened 28×28 pixel grid), and classification is based on projecting new test digits onto low-dimensional subspaces learned from training data.

🧠 Method Summary

1. **Training Phase**:
   - For each digit (0–9), training images are stacked into a matrix.
   - The SVD is computed for each digit-specific matrix.
   - The top `k` singular vectors (`U_k`) are used to define a low-dimensional subspace (digit "signature").

2. **Classification Phase**:
   - Each test digit is compared to all 10 digit subspaces.
   - The digit is assigned to the class whose subspace gives the **smallest residual** in a least-squares sense.

📊 Evaluation

The model is tested on 40,000 test digits using different values of `k` (number of singular vectors), and performance is reported as classification accuracy per digit.

📁 Files & Structure

- `TrainDigits.npy`, `TrainLabels.npy`: Training data and labels (10 classes).
- `TestDigits.npy`, `TestLabels.npy`: Test data and ground truth.
- `svd_classifier.py`: Main Python script implementing the classification algorithm.
- `plots/`: Figures of singular values and singular images for digits.
- `report.pdf`: Project report with discussion, code snippets, and results.
- `README.md`: This file.

🛠 Technologies Used -------------------

- Python (NumPy, Matplotlib)
- SVD via `numpy.linalg.svd`
- Efficient batch processing with matrix operations

Notes -------------------

- Only the first 400 training images per digit are used for training (due to time/memory constraints).
- The algorithm avoids solving explicit least-squares problems by using the closed-form residual formula:
  
  \[
  \text{residual} = \| (I - U_k U_k^T)\delta \|_2
  \]

- Singular values and "singular images" provide useful insights into the structure of digit classes.

Dataset -------------------

Feel free to use, study, and adapt this code for educational or research purposes.
