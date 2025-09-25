# Weather Classification using Machine Learning

A comprehensive machine learning project that implements and compares various classification models for weather type prediction using meteorological data.

## Project Overview

This project demonstrates a complete machine learning pipeline for multi-class weather classification. It compares traditional machine learning algorithms with deep learning approaches to identify the most effective method for predicting weather conditions.

## Key Results

### Model Performance Comparison

| Model | Accuracy | Weighted F1-Score |
|-------|----------|------------------|
| **SVM (RBF Kernel)** | **90.98%** | **91.00%** |
| LSTM | 90.83% | 90.83% |
| GRU | 90.80% | 90.80% |
| Gradient Boosting | 89.73% | 89.75% |
| Random Forest | 89.47% | 89.48% |
| SVM (Polynomial) | 88.26% | 88.30% |
| Logistic Regression | 85.61% | 85.56% |

## 🛠️ Technical Implementation

### Data Preprocessing
- **Dimensionality Reduction**: PCA with MLE (Maximum Likelihood Estimation) for optimal component selection
- **Feature Scaling**: StandardScaler for normalization
- **Sequential Data**: Time-series sequence creation for RNN models

### Models Implemented

#### Traditional Machine Learning
- Support Vector Machines (RBF and Polynomial kernels)
- Logistic Regression with L2 regularization
- Random Forest (Bagging ensemble)
- Gradient Boosting (Boosting ensemble)

#### Deep Learning
- **LSTM**: Bidirectional architecture with dropout regularization
- **GRU**: Gated Recurrent Unit with optimized hyperparameters

### Hyperparameter Optimization
- Random search with 20 iterations for neural networks
- Early stopping to prevent overfitting
- Cross-validation strategies

