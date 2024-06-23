# Medical Insurance Price Prediction

## Project Overview

This project aims to predict the price of medical insurance premiums using machine learning techniques in Python. By analyzing a dataset containing personal and insurance details, we seek to uncover insights and patterns that influence the cost of premiums.

## Features

- **Exploratory Data Analysis (EDA)**: Understand data distribution and relationships through visualizations.
- **Data Preprocessing**: Clean and prepare the data for modeling, including handling missing values and outliers.
- **Model Development**: Train and evaluate various machine learning models to find the best predictor for insurance premiums.

## Technologies Used

- **Python Libraries**:
  - `Pandas`: Data manipulation and analysis.
  - `Numpy`: Numerical computing.
  - `Matplotlib`/`Seaborn`: Data visualization.
  - `Scikit-learn`: Machine learning algorithms.
  - `XGBoost`: Advanced gradient boosting techniques.

## Dataset

The dataset contains 1338 entries with the following features:

- `age`: Age of the primary beneficiary.
- `sex`: Gender of the beneficiary.
- `bmi`: Body Mass Index.
- `children`: Number of children covered by the insurance.
- `smoker`: Smoking status.
- `region`: Residential area.
- `expenses`: Individual medical costs billed by health insurance.

## Steps to Run the Project

1. **Import Libraries and Load Dataset**:

   ```python
   import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   ```

2. **Data Preprocessing**:

   - Handle missing values and outliers.
   - Encode categorical features.
   - Normalize continuous features if necessary.

3. **Exploratory Data Analysis (EDA)**:

   - Visualize data distributions and relationships.
   - Identify correlations between features.

4. **Model Training**:

   - Split data into training and test sets.
   - Train multiple models (Linear Regression, Lasso, SVR, Random Forest, Gradient Boosting, XGBoost).
   - Evaluate models using cross-validation and select the best performing model.

5. **Prediction and Evaluation**:
   - Predict insurance costs on the test set.
   - Evaluate the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

## Conclusion

By following the steps outlined above, you can replicate the process of predicting medical insurance premiums using machine learning. The project demonstrates the importance of data preprocessing, exploratory analysis, and model evaluation in developing accurate predictive models.
