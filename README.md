# Face Clustering Using Gradient-Based Affinity and Spectral Clustering

This project demonstrates face clustering techniques using both K-Means and Spectral Clustering methods. It introduces a gradient-based affinity measure to improve clustering accuracy by leveraging image gradient and orientation features.

## Features

- **Gradient-Based Affinity Matrix:** Computes a similarity matrix based on image gradients and orientations to capture intrinsic structural information.
- **Downsampling for Efficiency:** Reduces image dimensions for faster computation while retaining essential features.
- **Clustering Algorithms:**
  - **K-Means Clustering:** Baseline clustering approach.
  - **Spectral Clustering:** Utilizes the gradient-based affinity matrix for improved performance.
- **Visualization:** Displays faces from the dataset under various conditions to provide insights into data variability.

## Requirements

To run the project, ensure you have the following dependencies installed:

- **Python**
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `scikit-learn`

## Workflow

1. **Data Loading:** Reads face images and metadata from a `.mat` file (`allFaces.mat`).
2. **Gradient Computation:** Calculates gradient magnitude and orientation for each face image.
3. **Affinity Matrix Construction:** Builds a similarity matrix using gradient and orientation information.
4. **Clustering:**
   - **K-Means:** Performs clustering using pixel intensities as features.
   - **Spectral Clustering:** Clusters faces using the gradient-based affinity matrix.
5. **Evaluation:** Computes clustering accuracy using the Adjusted Rand Index (ARI).
6. **Visualization:** Plots sample faces and results under varying conditions.

## Example Results

- **K-Means Clustering Accuracy:** 1.17%
- **Spectral Clustering with Gradient Affinity Accuracy:** 85.59%

These results highlight the effectiveness of the gradient-based affinity matrix in enhancing clustering performance.

## Applications

This project can be applied in:

- Facial recognition preprocessing
- Grouping similar faces for dataset organization
- Clustering tasks in image processing and analysis

## Example Dataset

The project processes face images stored in `allFaces.mat`, with clustering results demonstrating the impact of advanced affinity measures on accuracy.

---

Feel free to explore and adapt this project for your own clustering tasks!

more informatio about the algorithm is available here: 10.1109/CVPR.2003.1211332
