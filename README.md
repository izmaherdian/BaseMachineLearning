# Base Machine Learning Project

This repository contains basic machine learning implementations and mathematical concepts developed as part of the academic project at **Bandung Institute of Technology (ITB)**. The project covers essential topics ranging from evaluation metrics and logistic regression to deep learning (neural networks, CNNs, ResNet50) and dimensionality reduction/compression (PCA, SVD).

---

## 👥 Contributors
This project was co-authored by:
1. **Taufik Syahputra** (13321012)
2. **Izma Alhazmi Herdian** (13321027)
3. **Naufan Aurezan Mulyawan** (13321043)

---

## 📂 Project Structure
* 📄 **[Program.ipynb](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Program.ipynb)**: The main Jupyter Notebook containing all explanations, mathematical derivations, code implementations, and experiment visualizations.
* 🖼️ **[Cat.png](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Cat.png)**: Target image used for ResNet50 pre-trained image classification and Singular Value Decomposition (SVD) low-rank approximation.

---

## ❓ Notebook Questions & Problems Overview
Below is the systematic list of questions and problems addressed in the [Program.ipynb](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Program.ipynb) notebook:

### **Problem 1: Soft vs. Hard Classification Metrics**
* **Question/Task:** Explain and implement classification metrics for both soft classification (probabilistic outputs) and hard classification (discrete outputs).
* **Topics Covered:** 
  * *Soft metrics:* Soft Accuracy, Cross-Entropy Loss, Brier Score.
  * *Hard metrics:* Accuracy, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC), Jaccard Index.
  * Implement code functions for each metric and group them into a tabular evaluation format.

### **Problem 2: Evaluation Visualizations (Confusion Matrix & ROC)**
* **Question/Task:** Explain the theoretical concept of the *Confusion Matrix* and the *Receiver-Operating Characteristic (ROC)* curve alongside the *Area Under the Curve (AUC)* metric.

### **Problem 3: Binary Logistic Regression using JAX (Iris Dataset)**
* **Question/Task:** Load the Iris dataset, filter it to keep only two classes (`versicolor` and `virginica`), normalize the features, and implement a custom binary logistic regression model.
* **Topics Covered:**
  * Define a binary classifier using the sigmoid function.
  * Train the classifier utilizing automatic differentiation via JAX (`jax.grad`) to calculate gradients for weights and biases.
  * Plot training/testing loss and visualize decision boundaries.

### **Problem 4: Binary Logistic Regression on Hand-written Digits (Digits Dataset)**
* **Question/Task:** Apply the JAX-based binary logistic regression classifier developed in Problem 3 to classify hand-written digits 4 and 6 from the Scikit-learn Digits dataset.
* **Topics Covered:**
  * Standarize features and run custom batch training.
  * Visualize decision boundaries and plot images of the digits with their corresponding actual vs. predicted labels.

### **Problem 5: Multiclass Classification & Multilayer Perceptron (Fashion MNIST)**
* **Question/Task:** 
  1. Describe the Fashion MNIST dataset (number of classes, labels, shape, size, splits).
  2. Explain the concept of Multiclass Classification and the architecture of Multilayer Neural Networks (Input, Hidden, and Output layers).
  3. Define a standard training protocol for multiclass classification using a Multilayer Perceptron (MLP) without using convolutional, maxpooling, or other specialized layers.
  4. Define key hyperparameters (*learning rate*, *epochs*, *batch size*, *hidden nodes*, *activation functions*).
  5. Build and train an MLP using TensorFlow/Keras, and evaluate model performance under different hyperparameter settings (epochs: 1, 50, 100; activation: Tanh, ReLU; optimizer: SGD, Adam).

### **Problem 6: Deepening Neural Networks with CNNs (Fashion MNIST)**
* **Question/Task:** Improve the multiclass classifier from Problem 5 by introducing a Convolutional Layer (Conv2D), MaxPooling, Dropout layers, and additional Dense layers. Compare the classification accuracy against the standard MLP.

### **Problem 7: Pre-trained ResNet50 Image Classification**
* **Question/Task:** Explain the methodology behind the ResNet50 (Residual Network) architecture. Use a pre-trained ResNet50 model to classify the custom image [Cat.png](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Cat.png) and decode the top-3 predictions.

### **Problem 8: Principal Component Analysis (PCA) - Standardized vs. Unstandardized**
* **Question/Task:** Perform PCA on a synthetic cluster dataset containing 2 main features and 5 noisy features. Contrast the results of running PCA with and without data standardization (Z-score normalization) and explain how scaling affects principal components.

### **Problem 9: Singular Value Decomposition (SVD) and PCA Relationship**
* **Question/Task:** Mathematically explain the Singular Value Decomposition (SVD) theorem ($\mathbf{X} = \mathbf{U \Sigma V}^*$) and derive its exact relationship with Principal Component Analysis (PCA) and the covariance matrix.

### **Problem 10: Grayscale Image Compression via SVD**
* **Question/Task:** Convert the image [Cat.png](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Cat.png) to grayscale, decompose its matrix using SVD, and perform low-rank approximations at multiple ranks ($r = 2, 5, 10, 25, 60, 100, 200$).
* **Topics Covered:**
  * Plot the original, low-rank approximated, and difference images.
  * Measure and plot the percentage of information retained and the compression ratio as a function of rank $r$.

---

## 🛠️ Requirements & Installation
To run this project, make sure you have python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jax jaxlib pillow
```

---

## 🚀 How to Run the Notebook
1. Open your terminal in the project directory.
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
3. Open **[Program.ipynb](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Program.ipynb)** and run the cells sequentially.
