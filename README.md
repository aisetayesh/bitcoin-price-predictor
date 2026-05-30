# Bitcoin Price Movement Prediction

## Project Overview
This project predicts the next-day direction of Bitcoin closing price using historical market data and machine learning classification models.

## Main Features
- Data loading and initial analysis
- Price trend and distribution visualization
- Date-based feature extraction
- Feature engineering for prediction
- Model training and evaluation
- Comparison of multiple classifiers

## Models Used
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier

## Evaluation Metric
The project uses ROC AUC Score to evaluate model performance.

## Dataset
The dataset must be placed in the following path:

`./dataset/bitcoin.csv`

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run
1. Install the required libraries
2. Place the dataset in the dataset folder
3. Run the Python script

## Notes
This project is focused on predicting Bitcoin price movement direction, not exact price value.