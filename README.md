# 📈 Data Analysis and Machine Learning Project Collection

This repository serves as a professional portfolio showcasing a comprehensive collection of data analysis and machine learning projects, experiments, and techniques. It is structured to demonstrate proficiency in the complete data science workflow, from foundational data exploration and preprocessing to advanced predictive modeling, time series analysis, and model evaluation.

The entire collection is organized into three distinct learning and project modules: **OPTI**, **OPTII**, and **OPTIII**. In the II and III modules there is a folder called "Proyectos" containing all the knowledge across the different notebooks in each main folder.

---

## 📂 Project Structure and Learning Modules

The repository is divided into three main folders (`OPTI`, `OPTII`, `OPTIII`), each designed to build upon the skills and concepts introduced in the previous module.

### 1. OPTI: Foundational Data Exploration and Tools

This module focuses on fundamental data manipulation and exploratory data analysis (EDA) techniques. It is a collection of experiments designed for initial tool exploration and skill building.

* **Focus:** Learning to handle large datasets, confirming basic hypotheses, and mastering essential tools.
* **Key Techniques Covered:** Data indexing and selection (`df.reindex`), handling missing values (`df.dropna`), data type inspection (`df.dtypes`), basic statistical measures (`mean`, `std`), and data visualization (`matplotlib`).
* **Deliverables:** Various simple experiments and tutorials (no final projects).

### 2. OPTII: Regression, Advanced Preprocessing, and Predictive Modeling

This module transitions from foundational experiments to structured projects, introducing techniques for preparing data for modeling and building initial predictive systems.

* **Focus:** Combining regression methods with data visualization to understand data behavior, formulate hypotheses, and build reusable machine learning pipelines.
* **Key Techniques Covered:**
    * **Descriptive & Statistical Analysis:** Comprehensive exploration of complex datasets, identifying patterns, outliers, and data peculiarities.
    * **Data Cleaning & Preprocessing:** Data cleaning, structuring, handling missing values via **Imputation** (`SimpleImputer`), and applying advanced **Data Scaling** techniques (`StandardScaler`, `MinMaxScaler`) to approach Gaussian distribution.
    * **Feature Engineering:** Correlational analysis (`Df.corr()`, `pairwise_distances`) for variable selection and analysis of conditional probabilities.
    * **Pipeline Engineering:** Implementing reusable **Scikit-learn Pipelines** and specialized **ColumnTransformers** to process numerical and categorical data simultaneously.
    * **Model Deployment Prep:** Serialization of preprocessing pipelines using **joblib** and testing the loading/transformation with new, raw input data.

### 3. OPTIII: Model Comparison, Time Series Analysis, and Advanced ML

This module covers complex predictive challenges, focusing on rigorously evaluating model performance, hyperparameter optimization, and specialized techniques like time series analysis.

* **Focus:** Building and comparing multiple advanced predictive models, assessing their metrics, and tackling sequential data problems.
* **Key Techniques Covered:**
    * **Regression & Evaluation:** Implementing and evaluating regression models (**RandomForestRegressor**) using **Cross-Validation** and rigorous metrics like **MSE** (Mean Squared Error) and **RMSE** (Root Mean Squared Error).
    * **Hyperparameter Tuning:** Systematic optimization of model parameters using **GridSearchCV** to minimize error and ensure the best generalization capacity.
    * **Classification & Metrics:** Building classification models (**RandomForestClassifier**, **KNN**, **LogisticRegression**, **DecisionTree**) and assessing their performance with **Accuracy**, **Precision**, and **Recall**.
    * **Time Series Forecasting:** Analysis and modeling of sequential data (e.g., PM2.5 levels, stock prices), including identification of stationarity and seasonality.
    * **Time Series Models:** Utilizing advanced techniques like **ARIMA** (Autoregressive Integrated Moving Average) and implementing a **Persistence Model** (Naive) as a performance baseline.
    * **Advanced Data Transformation:** Converting a Time Series Regression problem into a **Binary Classification** task for alternative prediction modeling.

---

## 🛠️ Key Technologies and Libraries

The projects are primarily developed in Python, leveraging industry-standard libraries for the entire data science lifecycle.

| Category | Libraries / Tools |
| :--- | :--- |
| **Language** | Python |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (sklearn), `RandomForestClassifier`, `KNeighborsClassifier`, `LogisticRegression`, `RandomForestRegressor`, `DecisionTree` |
| **Time Series** | `statsmodels` (ARIMA, seasonal decomposition) |
| **Preprocessing & Pipelines** | `ColumnTransformer`, `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `joblib` (for serialization) |
| **Model Evaluation** | `GridSearchCV` (hyperparameter tuning), Cross-validation, **RMSE**, **MSE**, `accuracy_score`, `precision_score`, `recall_score` |
