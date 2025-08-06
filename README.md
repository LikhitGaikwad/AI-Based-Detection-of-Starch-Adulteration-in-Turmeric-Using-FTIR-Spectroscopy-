# AI-Based Detection of Starch Adulteration in Turmeric Using FTIR Spectroscopy 

Turmeric Starch Detection using FTIR Spectroscopy
This repository contains the complete machine learning pipeline for detecting starch adulteration in turmeric using Fourier-Transform Infrared (FTIR) spectroscopy data. The project aims to provide a robust and efficient method for quality control in the food industry.

# Table of Contents
1) Project Overview
2) Dataset
3) Preprocessing Techniques
4) Machine Learning Models
5) Results

# Project Overview
Turmeric is a widely used spice known for its health benefits and culinary applications. However, it is often susceptible to adulteration with cheaper substances like starch, which can compromise its quality and safety. FTIR spectroscopy offers a rapid and non-destructive method for analyzing the chemical composition of samples.

This project leverages FTIR spectral data to build and evaluate machine learning models capable of accurately predicting the adulteration percentage of starch in turmeric samples. The pipeline includes data loading, preprocessing, exploratory data analysis, model training, and evaluation.

# How FTIR Works – A Brief Overview 
FTIR (Fourier Transform Infrared) spectroscopy is an analytical technique used to identify 
and quantify chemical compounds based on how molecules absorb infrared (IR) light. When 
IR radiation is passed through a sample, specific wavelengths are absorbed depending on the 
molecular bonds present. These absorbed frequencies correspond to the vibrational modes of 
the molecules. 
The core of the FTIR instrument is the interferometer, which modulates the IR light and 
produces an interferogram—a signal representing light intensity as a function of the position 
of a moving mirror. This signal is then mathematically converted using Fourier Transform 
into an IR spectrum, showing absorbance vs. wavenumber (cm⁻¹). 
Each compound has a unique IR fingerprint, allowing FTIR to detect functional groups and 
chemical changes, including adulterants, in complex samples like turmeric powder. 
# Pellet Press Overview and FTIR Scanning 
To prepare the turmeric-starch mixtures for FTIR analysis, each 1 g sample was finely ground 
and homogenized. The prepared powders were then transferred to a pellet press unit. In this 
setup, a precise amount of the sample was compressed under high pressure to form a thin, 
compact disc (pellet). This ensures uniform thickness and optimal surface contact for infrared 
light transmission, minimizing scattering and enhancing spectral quality.  

# Spectral data collection and analysis 
Once the pellets were formed, they were carefully placed in the FTIR sample holder. Each 
sample underwent 32 scans in the mid-infrared range (4000 cm⁻¹ to 400 cm⁻¹) using the 
transmission mode. This provided high-resolution spectral data with improved signal-to-noise 
ratio. The collected spectra were saved for subsequent preprocessing and machine learning 
analysis to detect and quantify the level of starch adulteration.

The collected spectral data were processed using Jupyter Notebook, a web-based application 
for machine learning, utilizing Python 3.10.9. The following updated libraries were 
employed for data handling and analysis: NumPy 2.2.5, Pandas 2.2.3, Matplotlib 3.10.1, 
Seaborn 0.13.2, TensorFlow 2.19.0, and Scikit-Learn 1.6.1. These tools provided the 
computational backbone for preprocessing, visualization, feature extraction, and model 
development. The analysis was conducted on a system running Windows 11 Home Single 
Language equipped with an AMD Ryzen 7 5800H processor, 16 GB RAM, and NVIDIA 
GeForce RTX 3050 GPU, ensuring smooth execution of ML workflows. The raw FTIR data 
and corresponding adulteration percentages were compiled into an MS Excel file, referred to 
as the original dataset "X_raw" throughout the manuscript.

# Dataset
The dataset used in this project is named FTIR Data.csv. It contains:

Sample ID: Unique identifier for each turmeric sample.
Adulteration Percentage: The target variable, indicating the percentage of starch adulteration (e.g., 0%, 10%, 20%, 30%, 40%, 50% etc.).
Spectral Data: A large number of columns representing absorbance values at different wavenumbers (from 400 cm⁻¹ to 4000 cm⁻¹), which are the features for the machine learning models.

# Preprocessing Techniques
To enhance the quality of the FTIR spectral data and improve model performance, the following preprocessing techniques are applied:

1) Savitzky-Golay (SG) Smoothing:
Purpose: Reduces noise in the spectra while preserving the signal's shape.
Parameters: window_length=11, polyorder=2.
2) Standard Normal Variate (SNV):
Purpose: Corrects for light scattering and particle size effects by normalizing each spectrum to have a mean of zero and a standard deviation of one.
3) Extended Multiplicative Scatter Correction (EMSC):
Purpose: A more advanced scattering correction method that models and removes baseline shifts and multiplicative effects. A simplified version is implemented here.

# Machine Learning Models
The following regression models are explored to predict the adulteration percentage:

1) Linear Regression: A simple linear model to establish baseline performance.
2) Decision Tree Regressor: A non-linear model capable of capturing complex relationships in the data.
3) Support Vector Regressor (SVR): A powerful model effective in high-dimensional spaces.
4) XGBoost: Ensemble method, regularized boosting of decision trees.
5) Artificial Neural Network (ANN): Captures complex non-linear spectral
 pattern

# ANN Architecture
 Model: MLPRegressor with 2 hidden layers (64 & 32 neurons).
 ReLU activation in hidden layers; linear in output.
 Trained over 1000 iterations; random state = 42

 # Performance Evaluation
 Metrics: R², RMSE, MAE.
 Best models: ANN on SNV, ANN on Raw, and SG-preprocessed
 data.
 Worst: SVR on SNV and EMSC across models

 # Feature Selection
 Method: Recursive Feature Elimination (RFE) with SVR (linear).
 Top 15 wavenumbers selected from SNV-preprocessed data.
 Focused around 1600–1610 cm⁻¹, indicating key chemical shifts.
 # Feature Extraction
 Method: Principal Component Analysis (PCA) with z-score
 normalization.
 6 principal components retained, covering over 95% variance.
 Top 5 contributing wavenumbers identified per component.
 Helped reduce dimensionality while preserving interpretability

# Model Optimization via GridSearchCV
 Tuned ANN model on 6 datasets: X_raw, X_snv, X_rfe_raw,
 X_rfe_snv, X_pca_raw, X_pca_snv,.
 Grid parameters:
 Hidden layers: (32,), (64,), (64, 32)
 Alpha: 0.0001, 0.001
 Activation: ReLU
 Learning rate: constant
 Used 3-fold CV and 80:20 train-test split.
 Evaluated each config using R², RMSE, and MAE

# Best Model:
 1) Best Model: 
 ANN + SNV → R² = 0.887, RMSE = 1.68, MAE = 1.23
 2) Moderate Models:
 Raw + ANN and SG + ANN → R² ≈ -0.504
 3) Worst Performers:
 SVR + SNV → R² = –35.80, RMSE = 30.33
 EMSC + ANN → R² = –3.31, RMSE = high
 EMSC + SVR → R² = –34.80
 XGBoost:
 R² = 0.31, RMSE = 14.14
 Less effective than ANN; possibly due to limited tuning or inability to model
 spectral complexity

# All model Performances
<img width="1152" height="612" alt="image" src="https://github.com/user-attachments/assets/21e19ddc-ffdd-4f45-ac07-b4287e6a0e36" />

 

