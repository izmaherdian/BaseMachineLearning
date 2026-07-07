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

## ❓ Problems & Results Overview
Below is the systematic list of problems, tasks, and results implemented in the **[Program.ipynb](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Program.ipynb)** notebook:

### **Problem 1: Soft vs. Hard Classification Metrics**
* **The Problem:** Define and implement classification evaluation metrics for both soft classification (probabilistic outputs) and hard classification (discrete outputs) using raw Numpy/JAX.
* **The Results:**
  * Created code definitions for:
    * *Soft metrics:* Soft Accuracy, Cross-Entropy Loss, Brier Score.
    * *Hard metrics:* Accuracy, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC), Jaccard Index.
  * Formatted the metrics output inside a pandas DataFrame via a custom `metrics(y, yhat)` function for unified evaluation.

### **Problem 2: Evaluation Visualizations (Confusion Matrix & ROC)**
* **The Problem:** Provide theoretical explanations of the *Confusion Matrix* and the *Receiver-Operating Characteristic (ROC)* curve alongside the *Area Under the Curve (AUC)* metric.
* **The Results:**
  * Documented how Confusion Matrices show performance (True Positive, False Positive, True Negative, False Negative) and facilitate metric calculations.
  * Demonstrated that ROC plots the True Positive Rate (TPR) against the False Positive Rate (FPR), and AUC summarizes classifier performance (where an AUC of 0.5 is equivalent to random guessing and 1.0 is a perfect classifier).

### **Problem 3: Binary Logistic Regression using JAX (Iris Dataset)**
* **The Problem:** Classify the Iris dataset into two categories (`versicolor` as label 0 and `virginica` as label 1) using normalized features and a custom binary logistic regression model optimized with JAX auto-differentiation.
* **The Results:**
  * Implemented gradient descent where gradients of weights and biases are automatically computed using `jax.grad`.
  * **Evaluation Outcome:** The model achieved **excellent classification performance with Accuracy and Precision > 90%**, indicating that the linear decision boundary effectively separates the two iris classes.

### **Problem 4: Binary Logistic Regression on Hand-written Digits (Digits Dataset)**
* **The Problem:** Classify hand-written digit images of `4` and `6` from the Scikit-learn Digits dataset using the JAX-based binary logistic regression model.
* **The Results:**
  * Trained the model on normalized $8 \times 8$ grayscale digit features.
  * **Evaluation Outcome:** The model achieved a **good baseline classification performance (Accuracy & Precision > 70%)**. The notebook notes that adjusting epochs and batch sizes can optimize this, but if performance plateaus, a standard Sigmoid-based linear model might be sub-optimal for complex non-linear digit variations.

### **Problem 5: Multiclass Classification & Multilayer Perceptron (Fashion MNIST)**
* **The Problem:** Describe the Fashion MNIST dataset, explain multiclass classification, outline neural network architectures, and train a basic Multilayer Perceptron (MLP) (without Conv/Pooling layers) while experimenting with hyperparameters.
* **The Results:**
  * Created a simple MLP with 1 hidden layer (50 Tanh/ReLU units) and an output layer (10 Softmax units) trained via TensorFlow/Keras.
  * **Hyperparameter Experimentation Outcomes:**
    * **Default** (`epochs=50`, `activation='tanh'`, `optimizer='SGD'`): **87.43%** test accuracy.
    * **Low Epochs** (`epochs=1`, `activation='tanh'`, `optimizer='SGD'`): **81.69%** (underfitted due to too few iterations).
    * **High Epochs** (`epochs=100`, `activation='tanh'`, `optimizer='SGD'`): **87.56%** (no significant gain over 50 epochs).
    * **ReLU Activation** (`epochs=50`, `activation='relu'`, `optimizer='SGD'`): **87.42%** (similar to Tanh).
    * **Adam Optimizer** (`epochs=50`, `activation='tanh'`, `optimizer='Adam'`): **86.05%** (slightly lower than SGD, indicating default SGD optimization was more suitable).

### **Problem 6: Deepening Neural Networks with CNNs (Fashion MNIST)**
* **The Problem:** Enhance the classification model by adding a Convolutional Layer (Conv2D), MaxPooling, Dropout layers, and additional Dense layers. Compare the performance against the standard MLP model.
* **The Results:**
  * Implemented and trained a CNN model for 50 epochs.
  * **Evaluation Outcome:** The test accuracy **increased significantly to 91.19%**. The additional convolutional layers allowed the model to extract spatial image features more effectively, while Dropout prevented overfitting, resulting in a much more robust classifier.

### **Problem 7: Pre-trained ResNet50 Image Classification**
* **The Problem:** Explain the methodology behind the ResNet50 (Residual Network) architecture and use a pre-trained ResNet50 model to classify the custom image [Cat.png](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Cat.png).
* **The Results:**
  * Provided a breakdown of ResNet50's 50 layers, parameter size, and skip-connections.
  * Successfully preprocessed the target image and classified it using weights pre-trained on ImageNet, outputting the top-3 predictions.

### **Problem 8: PCA (Principal Component Analysis) - Standardized vs. Unstandardized**
* **The Problem:** Perform PCA on a synthetic cluster dataset containing 2 main features and 5 noisy features to contrast the effects of standardization vs. non-standardization.
* **The Results:**
  * **Standardized PCA:** The variance was spread relatively evenly across all principal components due to Z-score scaling.
  * **Unstandardized PCA:** PC1 and PC2 captured **95% of total variance** (PC1: 58%, PC2: 37%), which aligned perfectly with the original two-dimensional cluster data before adding random noise.
  * **Conclusion:** When features share the same units and are measured by the same instrument, unstandardized PCA preserves the true original cluster variance better.

### **Problem 9: Singular Value Decomposition (SVD) and PCA Relationship**
* **The Problem:** Mathematically formulate the Singular Value Decomposition (SVD) theorem ($\mathbf{X} = \mathbf{U \Sigma V}^*$) and derive its exact relationship with Principal Component Analysis (PCA) and the covariance matrix.
* **The Results:**
  * Mathematically proved that the sample covariance matrix $\mathbf{Q} = \mathbf{X}^\mathrm{T}\mathbf{X}$ relates directly to SVD singular values $\mathbf{\Sigma}$ and right-singular vectors $\mathbf{W}$.
  * Confirmed that SVD-based PCA is computationally more efficient than calculating the covariance matrix directly.

### **Problem 10: Grayscale Image Compression via SVD**
* **The Problem:** Convert the image [Cat.png](file:///run/media/izmaherdian/Windows-SSD/Izma_S2_InstrumentasiKontrol_ITB/Akademik/Project S1/BaseMachineLearning/Cat.png) to grayscale and perform low-rank SVD approximations to compress it. Evaluate the compression ratios and information retained at ranks $r = 2, 5, 10, 25, 60, 100, 200$.
* **The Results:**
  * Computed SVD matrices and reconstructed images at various rank thresholds.
  * **Compression vs. Quality Outcome:**
    * Even at low ranks (e.g., $r = 25$ or $r = 60$), the model retained a high percentage of visual information while achieving high compression rates.
    * Plotted a curve of information retained (%) as a function of Rank $r$ to visually analyze the compression-tradeoff.

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
