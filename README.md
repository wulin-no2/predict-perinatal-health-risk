# Predicting Maternal Health Risk

This project uses machine learning models to predict maternal health risks (high, low, or mid) based on various health indicators such as Age, SystolicBP, DiastolicBP, Blood Sugar (BS), Body Temperature, and Heart Rate. We have applied and compared different machine learning models, including Decision Tree and K-Nearest Neighbors (KNN), with hyperparameter tuning for improved performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Models Used](#models-used)
- [Performance Evaluation](#performance-evaluation)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Results](#results)


## Project Overview

This project aims to predict maternal health risks using different classification models. The dataset includes information on six health indicators, and the task is to classify patients into risk categories. Models are evaluated based on accuracy, precision, recall, and F1-score.

## Models Used

1. **Decision Tree**
   - A tree-based model that splits the dataset based on feature values to make predictions.
   
2. **Tuned Decision Tree**
   - GridSearchCV was used to optimize hyperparameters (e.g., `max_depth`, `min_samples_split`).

3. **K-Nearest Neighbors (KNN)**
   - A similarity-based model where the classification is made by a majority vote of the nearest neighbors.

4. **Tuned KNN**
   - Hyperparameters such as the number of neighbors (`n_neighbors`), distance metrics, and weighting schemes were tuned using GridSearchCV.

## Performance Evaluation

Model performance was evaluated using:
- **Accuracy**: The percentage of correctly classified instances.
- **Confusion Matrix**: A matrix showing the distribution of true vs. predicted classes.
- **Precision, Recall, and F1-Score**: Metrics used for evaluating classification models, especially for imbalanced data.

## Installation

To run the project, you'll need Python 3.x and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Use

1. Clone this repository

2. Navigate to the project folder:

```bash
cd predict-perinatal-health-risk
```

3. Run the Jupyter notebook to train and evaluate the models:

```bash
jupyter notebook predict_perinatal_health_risk.ipynb
```

## Results
### Accuracy Comparisons:

| Model               | Accuracy |
|---------------------|----------|
| Decision Tree        | 82.75%   |
| Tuned Decision Tree  | 83.74%   |
| KNN                 | 66.99%   |
| Tuned KNN           | 80.30%   |





