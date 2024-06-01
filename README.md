# Stock Market Price Prediction using LSTM with Real-Time Data

## Overview

This project utilizes Long Short-Term Memory (LSTM) neural networks to predict stock market prices, integrating real-time data from Yahoo Finance. By leveraging LSTM models, the project aims to analyze historical stock data to forecast future prices, providing valuable insights for investors and traders.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Acquisition](#data-acquisition)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Results](#results)
8. [Technologies Used](#technologies-used)
9. [Contact](#contact)
10. [References](#references)
11. [Project Link](#project-link)

## Introduction

This project focuses on predicting stock market prices using LSTM neural networks, a type of recurrent neural network (RNN) well-suited for sequential data analysis. Real-time data from Yahoo Finance is utilized to train and validate the LSTM model, enabling accurate predictions of stock prices over time.

## Data Acquisition

- **Source:** Real-time stock market data is obtained from Yahoo Finance using the Yahoo Finance API.
- **Data:** Historical stock prices, including open, high, low, and closing prices, along with trading volume, are collected for analysis.

## Data Preprocessing

- **Scaling:** The acquired data is preprocessed and scaled using MinMaxScaler to normalize the features within a specific range, enhancing model performance.
- **Sequence Generation:** Sequences of input-output pairs are created to train the LSTM model, with each sequence representing a window of historical stock prices.

## Model Architecture

The LSTM model architecture consists of multiple layers, including LSTM layers with varying units, dropout layers for regularization, and a dense output layer. This architecture enables the model to effectively capture temporal dependencies in the stock price data.

## Model Training

### Steps:

1. **Data Splitting:** The dataset is split into training, validation, and testing sets.
2. **Model Initialization:** Initialize the LSTM model with the desired architecture.
3. **Model Compilation:** Compile the model with appropriate loss function and optimizer.
4. **Model Training:** Train the LSTM model using the training data.
5. **Early Stopping:** Implement early stopping to prevent overfitting during training.
6. **Hyperparameter Tuning:** Fine-tune model hyperparameters for optimal performance.

## Model Evaluation

- **Validation:** The trained model's performance is evaluated on validation data using mean squared error (MSE) as the primary metric.
- **Testing:** The final model is tested on a separate test set to assess its generalization performance and accuracy in predicting unseen data.

## Results

- **Training Accuracy:** The LSTM model achieves a training accuracy of 79% (MSE: 0.196), demonstrating its ability to learn from historical stock data.
- **Testing Accuracy:** The model achieves a testing accuracy of 44.6% (MSE: 0.101), indicating its capability to make predictions on unseen data.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Yahoo Finance API

## Contact

For inquiries or feedback, feel free to reach out:
- [Email](mailto:your.email@example.com)
- [LinkedIn](https://www.linkedin.com/in/your-profile/)

## References

- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## Project Link

For further details and access to the project repository, visit [this link](https://github.com/muadrahman/Stockmarket-prediction).
