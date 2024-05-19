# Stock Price Prediction Project

## Overview

This project aims to predict stock prices using machine learning techniques. We perform data analysis and feature engineering to improve the accuracy of our models. The project utilizes three different machine learning algorithms: Linear Regression, Support Vector Classification (SVC), and XGBoost Classifier. The test data used in this project is `tesla.csv`.

## Project Structure

The project consists of the following steps:

1. **Data Analysis**
2. **Feature Engineering**
3. **Model Training and Testing**

## Data Analysis

In the data analysis phase, we explore the dataset to understand the relationships between different features. This includes visualizing the data, checking for missing values, and understanding the distribution of the data.

## Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve the performance of the machine learning models. This may include scaling features, creating lag features, or extracting date-related features from the timestamps.

## Machine Learning Models

### 1. Linear Regression
Linear Regression is a simple and commonly used algorithm for regression problems. It assumes a linear relationship between the input features and the target variable.

### 2. Support Vector Classification (SVC)
Support Vector Classification is a classification algorithm that finds the hyperplane that best separates the data into different classes. Even though SVC is generally used for classification tasks, it can also be adapted for regression problems.

### 3. XGBoost Classifier
XGBoost is an optimized gradient boosting algorithm that is highly efficient and accurate. It is used for both regression and classification tasks.

## Model Training and Testing

We train and test our data on the three models mentioned above. Each model provides different accuracies based on its inherent characteristics and the feature set used. The test data used is `tesla.csv`.

## Results

The performance of the models is evaluated using various metrics. Here are the accuracies obtained from each model:

- **Linear Regression**: Accuracy - [insert accuracy]
- **Support Vector Classification**: Accuracy - [insert accuracy]
- **XGBoost Classifier**: Accuracy - [insert accuracy]

## Conclusion

This project demonstrates the application of different machine learning algorithms for stock price prediction. By comparing the performance of Linear Regression, Support Vector Classification, and XGBoost Classifier, we can determine which model is best suited for this task.

## Files in the Repository

- `main.ipynb`: Jupyter Notebook containing the entire project code.
- `tesla.csv`: The dataset used for testing.
- `README.md`: This file providing an overview of the project.

## How to Run the Project

1. Clone the repository.
2. Install the required libraries.
3. Open the `notebook.ipynb` file in Jupyter Notebook.
4. Execute the cells to run the data analysis, feature engineering, and model training/testing steps.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

Install the required libraries using pip:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn
