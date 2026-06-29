# HSC Result Prediction

This repository contains a machine learning project designed to predict the Higher Secondary Certificate (HSC) exam results (GPA) of students in Bangladesh. By analyzing demographic, socioeconomic, and prior academic features, the models aim to identify key determinants of student performance and predict final GPAs with high accuracy.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
4. [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
5. [Machine Learning Models](#-machine-learning-models)
6. [Model Evaluation & Results](#-model-evaluation--results)
7. [Project Structure](#-project-structure)
8. [Installation & Setup](#-installation--setup)

---

## 🔍 Project Overview

Academic performance is influenced by a complex interplay of demographic, family, socioeconomic, and personal factors. This project utilizes Python's data science ecosystem (`pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`) to:
* Impute missing data and encode categorical features through automated Pipelines.
* Scale and normalize numerical attributes to optimize gradient descent performance.
* Train and evaluate regression models, specifically **Linear Regression** and **Stochastic Gradient Descent (SGD) Regressor**.
* Validate model robustness using 5-fold cross-validation.

---

## 📊 Dataset Description

The dataset used is `bangladesh_student_performance.csv`, containing **2,018 rows** and **16 columns**. 

### Feature Details:
* **Academic Scores (GPA Scale 2.0 to 5.0):**
  * `ssc_result`: Secondary School Certificate (SSC) GPA (Feature)
  * `hsc_result`: Higher Secondary Certificate (HSC) GPA (**Target Variable**)
* **Demographics & Environment:**
  * `gender`: Student gender (`M` for Male, `F` for Female)
  * `age`: Age of the student (ranges from 17 to 19)
  * `address`: Residential location type (`Urban` or `Rural`)
* **Family Background:**
  * `famsize`: Family size category (`GT3` for greater than 3 members, `LE3` for less than or equal to 3 members)
  * `Pstatus`: Parents' cohabitation status (`Together` or `Apart`)
  * `M_Edu` / `F_Edu`: Mother's / Father's education level (numeric scale from `0` to `4`)
  * `M_Job` / `F_Job`: Mother's / Father's occupation category (e.g., `Teacher`, `Health`, `Services`, `At_home`, `Farmer`, `Business`, `Other`)
* **Personal & Social Factors:**
  * `relationship`: Whether the student is in a romantic relationship (`Yes` or `No`)
  * `smoker`: Student's smoking status (`Yes` or `No`)
  * `time_friends`: Rated time spent hanging out with friends (numeric scale from `1` to `5`)
  * `tuition_fee`: Tuition fee structure/amount paid (numeric value)
* **Metadata:**
  * `date`: The entry date of the record (Dropped during model training)

---

## 📈 Exploratory Data Analysis (EDA)

The [hsc_result_prediction.ipynb](file:///c:/Users/jalis/Desktop/ml/hsc_result_prediction/hsc_result_prediction.ipynb) notebook carries out several visualization tasks:
1. **Feature Distributions:** Distribution of numerical attributes (age, education level, tuition fees, time spent with friends).
2. **Class Balance:** Visualizing the proportions of gender, family size, and residential areas (Urban vs. Rural).
3. **Target Analysis:** Plotting the distribution of the target variable `hsc_result` (which displays a left-skewed shape peaking around GPA 3.0 - 4.5).
4. **SSC vs. HSC GPA Correlation:** A scatter plot confirming a very strong positive correlation between prior SSC results and final HSC results.
5. **Correlation Heatmap:** A Pearson correlation coefficient matrix computed over the numerical columns to check for multi-collinearity.

---

## 🛠️ Data Preprocessing Pipeline

The machine learning models ingest raw data handled by a modular pipeline via Scikit-Learn's `Pipeline` and `ColumnTransformer`:

* **Numerical Pipeline:**
  1. **Imputation:** Median values are filled for any missing coordinates via `SimpleImputer(strategy="median")`.
  2. **Scaling:** Features are standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
* **Categorical Pipeline:**
  1. **Imputation:** Missing values are imputed with the most frequent value using `SimpleImputer(strategy="most_frequent")`.
  2. **Encoding:** Categorical labels are converted to one-hot binary vectors using `OneHotEncoder(handle_unknown="ignore")`.
* **Feature Dropping:** The `date` column and the target column `hsc_result` are separated from the training set.
* **Data Split:** Split into an 80% training set and a 20% testing set using a seed of `random_state=42`.

---

## 🤖 Machine Learning Models

We implement two regression models to predict student GPA:

### 1. Linear Regression Pipeline
An Ordinary Least Squares (OLS) regression model. 
* **Implementation:** Wrapped in a unified pipeline with the `preprocessor`.

### 2. Stochastic Gradient Descent (SGD) Regressor
A linear model fitted by minimizing the loss function via SGD optimization. Extremely helpful for large datasets and regularized learning.
* **Loss Function:** Squared Error (`loss="squared_error"`)
* **Regularization:** L2 Penalty (Ridge Regression, `penalty="l2"`)
* **Regularization Strength ($\alpha$):** 0.0001
* **Initial Learning Rate ($\eta_0$):** 0.001 (Constant schedule, `learning_rate="constant"`)
* **Max Iterations:** 3000

---

## 📊 Model Evaluation & Results

The models are evaluated on both training and test subsets using **Coefficient of Determination ($R^2$)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**.

### Metric Summary Table:

| Model | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.9466** | **0.9459** | **0.1424** | **0.1114** |
| **SGD Regressor** | 0.9447 | 0.9453 | 0.1432 | 0.1116 |

### Cross-Validation
To verify the generalizability of the pipeline, **5-fold Cross-Validation** was performed using $R^2$ scoring on the entire dataset:
* **$R^2$ Scores per fold:** `[0.9482, 0.9530, 0.9390, 0.9432, 0.9418]`
* **Average Cross-Validation $R^2$ Score:** **0.9450**

The model exhibits high performance (~94.5% variance explained) with extremely low prediction errors (around 0.11 GPA points absolute deviation), without showing signs of overfitting.

---

## 📁 Project Structure

```
hsc_result_prediction/
│
├── bangladesh_student_performance.csv  # Raw student dataset
├── hsc_result_prediction.ipynb         # EDA, pipeline building, and model evaluation
└── README.md                           # Project documentation (this file)
```

---

## 🚀 Installation & Setup

Follow these steps to run the notebook locally:

### 1. Clone the repository / Open the folder
Navigate to the directory containing the project:
```bash
cd hsc_result_prediction
```

### 2. Install Dependencies
Ensure you have Python installed (preferably version 3.8 or above). Install the required packages using `pip`:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

### 3. Launch Jupyter Notebook
Run the following command to start the Jupyter interface:
```bash
jupyter notebook
```
Open `hsc_result_prediction.ipynb` in the browser interface, and run all cells to reproduce the exploratory plots and model metrics.
