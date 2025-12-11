# 📈 Data Analysis and Machine Learning Project Collection

This repository serves as a professional portfolio showcasing a comprehensive collection of data analysis and machine learning projects, experiments, and techniques. It is structured to demonstrate proficiency in the complete data science workflow, from foundational data exploration and preprocessing to advanced predictive modeling, time series analysis, and model evaluation.

The entire collection is organized into three distinct learning and project modules: **OPTI**, **OPTII**, and **OPTIII**.

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
    * **Data Preprocessing:** Advanced scaling methods (`StandardScaler`, `MinMaxScaler`), correlation analysis (`Df.corr()`, `pairwise_distances`), and using **Scikit-learn Pipelines** and **Column Transformers** for robust numerical and categorical data transformation.
    * **Descriptive Analysis:** Comprehensive statistical and visual exploration of complex datasets (e.g., IT salaries in Europe).
    * **Pipeline Engineering:** Serializing and reusing preprocessing pipelines using `joblib` for model deployment and testing with new data.

### 3. OPTIII: Model Comparison, Time Series Analysis, and Advanced ML

This module covers complex predictive challenges, focusing on rigorously evaluating model performance, hyperparameter optimization, and specialized techniques like time series analysis.

* **Focus:** Building and comparing multiple advanced predictive models, assessing their metrics, and tackling sequential data problems.
* **Key Techniques Covered:**
    * **Regression Modeling:** Building and optimizing models like **RandomForestRegressor** to predict continuous variables (e.g., car fuel consumption (MPG)), evaluated via **MSE** and **RMSE**.
    * **Classification & Evaluation:** Implementing and comparing various classification algorithms (`RandomForestClassifier`, `KNN`, `LogisticRegression`) using rigorous metrics (`Accuracy`, `Precision`, `Recall`) and cross-validation to manage bias and variance.
    * **Time Series Analysis:** Applying specialized methods (like **ARIMA** and the **Persistence Model**) to forecast sequential data (e.g., PM2.5 pollution and stock prices). This includes converting a time series regression problem into a **binary classification task**.

---

## 🛠️ Key Technologies and Libraries

The projects are primarily developed in Python, leveraging industry-standard libraries for the entire data science lifecycle.

| Category | Libraries / Tools |
| :--- | :--- |
| **Language** | Python |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (sklearn), `RandomForestClassifier`, `KNeighborsClassifier`, `LogisticRegression` |
| **Time Series** | `statsmodels` (ARIMA, seasonal decomposition) |
| **Preprocessing & Pipelines** | `ColumnTransformer`, `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `joblib` (for serialization) |
| **Model Evaluation** | `GridSearchCV` (hyperparameter tuning), Cross-validation, **RMSE**, **MSE**, `accuracy_score`, `precision_score`, `recall_score` |
