# Breast Cancer Treatment Response Prediction

This repository contains the complete machine learning workflow for predicting two key outcomes in breast cancer treatment: Pathological Complete Response (pCR) and Relapse-Free Survival (RFS). The project leverages a dataset combining clinical patient data and MRI-based radiomic features to build and evaluate predictive models for both classification and regression tasks.

## Project Overview

The project is divided into two distinct modeling tasks, each addressed in its own Jupyter Notebook:

1.  **pCR Prediction (Classification)**: A binary classification task to predict whether a patient will achieve a Pathological Complete Response (`pCR (outcome)`). The entire pipeline for this task is detailed in `Classification.ipynb`.
2.  **RFS Prediction (Regression)**: A regression task to predict the Relapse-Free Survival time in days (`RelapseFreeSurvival (outcome)`). The comprehensive workflow for this is in `Regression.ipynb`.

## Repository Structure

```
.
├── Classification.ipynb            # Notebook for pCR classification task.
├── Regression.ipynb                # Notebook for RFS regression task.
├── TrainDataset2025.xls            # Training dataset with clinical and radiomic features.
├── FinalTestDataset2025.xls        # Unlabeled test dataset for final predictions.
├── FinalTestPCR.csv                # Generated predictions for the pCR task.
└── FinalTestRFS.csv                # Generated predictions for the RFS task.
```

## Methodology

### 1. pCR Classification (`Classification.ipynb`)

This notebook implements an end-to-end pipeline to predict pCR.

-   **Target Variable**: `pCR (outcome)` (Binary: 1 for response, 0 for no response).
-   **Data Preprocessing**:
    -   Missing values, encoded as `999`, are identified and imputed.
    -   A custom `OutlierCapper` transformer handles extreme values in numerical features.
    -   A `ColumnTransformer` pipeline is used to apply different preprocessing steps to numerical, ordinal, and binary features, including scaling and one-hot encoding.
-   **Handling Class Imbalance**: The dataset is imbalanced. This is addressed by using `SMOTE` (Synthetic Minority Over-sampling Technique) within the pipeline and evaluating models based on `balanced_accuracy`.
-   **Model Selection**:
    -   Several baseline models were evaluated, including CatBoost, LightGBM, XGBoost, and MLPClassifier.
    -   The `MLPClassifier` combined with SMOTE demonstrated the most robust performance.
-   **Hyperparameter Tuning**: `RandomizedSearchCV` was used to find the optimal hyperparameters for the final MLP model.
-   **Final Model**: A tuned `MLPClassifier` with SMOTE, achieving a cross-validated balanced accuracy of **0.639**.

### 2. RFS Regression (`Regression.ipynb`)

This notebook details a systematic approach to predict RFS.

-   **Target Variable**: `RelapseFreeSurvival (outcome)` (Continuous).
-   **Data Preprocessing**:
    -   Compared multiple imputation strategies (`SimpleImputer`, `KNNImputer`) and scaling methods (`StandardScaler`, `RobustScaler`).
    -   The final preprocessing pipeline consists of `SimpleImputer` with a 'median' strategy followed by `RobustScaler`.
-   **Feature Selection**:
    -   To combat the "curse of dimensionality" (118 features for 400 samples), several feature selection techniques were evaluated: Mutual Information, Lasso, and Random Forest Importance.
    -   `Random Forest Importance` was selected, reducing the feature set to the **top 30 features** while improving model performance. The features `ER`, `HER2`, and `Gene` were mandatorily retained.
-   **Model Selection and Tuning**:
    -   A wide range of regression models (Ridge, Lasso, SVR, RandomForest, etc.) were benchmarked.
    -   `RandomForestRegressor` was identified as the top-performing model.
    -   A two-step hyperparameter tuning process (`RandomizedSearchCV` followed by `GridSearchCV`) was applied to optimize the model.
-   **Final Model**: A tuned `RandomForestRegressor` trained on 30 selected features, achieving a cross-validated Mean Absolute Error (MAE) of **20.33**.


### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn "imbalanced-learn" catboost xgboost lightgbm openpyxl
