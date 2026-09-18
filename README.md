# Credit Card Default Prediction using Logistic Regression

## 📌 Project Overview

This project focuses on predicting whether a credit-card customer will **default on their next payment** using Machine Learning.

The project uses **Logistic Regression** as the main classification algorithm and applies techniques such as data cleaning, outlier detection, feature scaling, cross-validation, GridSearchCV, and RandomizedSearchCV.

The final model achieved **80.91% accuracy** according to the results recorded in the notebook.

---

## 🎯 Objective

The main objective of this project is to build a machine learning classification model that can predict:

* **0** → Customer does not default
* **1** → Customer defaults on the next payment

This can help in analyzing customer payment behavior and identifying customers who may have a higher risk of defaulting.

---

## 📂 Dataset

The project uses the **Default of Credit Card Clients** dataset.

The dataset contains information about credit-card customers, including customer financial and demographic attributes.

The target column is:

```text
default payment next month
```

The `ID` column was removed because it was not used as a feature for prediction.

---

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🛠️ Project Workflow

### 1. Import Libraries

The following Python libraries were used:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn was used for machine learning, preprocessing, model evaluation, cross-validation, and hyperparameter tuning.

---

### 2. Load Dataset

The dataset was loaded using Pandas:

```python
df = pd.read_csv("default of credit card clients.csv")
```

---

### 3. Data Understanding

The dataset was explored using:

```python
df.head()
df.shape
df.info()
df.describe()
df.columns
```

These operations were used to understand the dataset structure, dimensions, data types, and statistical information.

---

### 4. Data Cleaning

Missing values and duplicate records were checked:

```python
df.isnull().sum()
df.duplicated().sum()
```

The `ID` column was removed:

```python
df = df.drop('ID', axis=1)
```

---

### 5. Outlier Detection

The **Interquartile Range (IQR)** method was used to identify potential outliers.

The calculation was:

```python
Q1 = num_col.quantile(0.25)
Q3 = num_col.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

The notebook notes that most detected outliers were considered valid values, so they were **not removed**.

---

## 📊 Exploratory Data Analysis

A bar chart was created to compare customers who defaulted and customers who did not:

```python
sns.countplot(
    x='default payment next month',
    data=df
)
```

The visualization shows that the dataset contains more customers who did not default than customers who defaulted.

---

## 🔀 Independent and Dependent Variables

The target variable was separated from the input features:

```python
X = df.drop(['default payment next month'], axis=1)

y = df['default payment next month']
```

Where:

* `X` → Independent variables/features
* `y` → Dependent variable/target

---

## ✂️ Train-Test Split

The dataset was divided into training and testing datasets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

This creates:

* **75% Training Data**
* **25% Testing Data**

---

## 📏 Feature Scaling

Standardization was performed using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

The scaler was fitted only on the training data and then used to transform the test data.

---

# 🤖 Machine Learning Model

## Logistic Regression

Logistic Regression was used as the primary classification algorithm.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train_scaled, y_train)
```

Predictions were generated using:

```python
y_pred = model.predict(x_test_scaled)
```

---

## 📈 Model Evaluation

The model was evaluated using:

### Accuracy

```python
accuracy_score(y_test, y_pred)
```

### Classification Report

```python
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

```python
print(confusion_matrix(y_test, y_pred))
```

The notebook reports an accuracy of:

**80.91%**

---

# 🔄 5-Fold Cross-Validation

5-Fold Cross-Validation was used to evaluate the Logistic Regression model across multiple splits of the training data.

```python
cv_score = cross_val_score(
    model,
    x_train_scaled,
    y_train,
    cv=5,
    scoring='accuracy'
)
```

This provides multiple accuracy scores instead of relying only on one train-test split.

---

# ⚙️ Hyperparameter Tuning

## GridSearchCV

GridSearchCV was used to search through different Logistic Regression hyperparameter combinations.

Parameters included:

* `penalty`
* `solver`
* `max_iter`
* `C`

Example:

```python
grid = GridSearchCV(
    estimator=model,
    param_grid=parameter,
    cv=cv,
    n_jobs=-1
)

grid.fit(x_train_scaled, y_train)
```

The best parameters were obtained using:

```python
grid.best_params_
```

The best cross-validation score was obtained using:

```python
grid.best_score_
```

The tuned model was then evaluated on the test data.

---

# 🎲 RandomizedSearchCV

RandomizedSearchCV was also used for hyperparameter tuning:

```python
rs = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter,
    cv=5,
    scoring='accuracy'
)

rs.fit(x_train_scaled, y_train)
```

The best parameters were obtained using:

```python
rs.best_params_
```

The best cross-validation score was obtained using:

```python
rs.best_score_
```

---

# 📊 Model Comparison

The notebook compares:

1. Logistic Regression
2. Logistic Regression with GridSearchCV
3. Logistic Regression with RandomizedSearchCV

The reported accuracy results were:

| Approach            | Accuracy |
| ------------------- | -------: |
| Logistic Regression |   80.91% |
| GridSearchCV        |   80.91% |
| RandomizedSearchCV  |   80.91% |

According to the notebook results, all three approaches produced the same reported accuracy.

---

# 🏁 Conclusion

In this project, credit-card customer data was cleaned and prepared for machine learning. Logistic Regression was used to predict whether a customer would default on their next payment.

The notebook reports **80.91% accuracy** for the Logistic Regression model. GridSearchCV and RandomizedSearchCV also produced the same reported accuracy.

The project demonstrates the complete machine-learning workflow, including:

* Data understanding
* Data cleaning
* Outlier detection
* Exploratory Data Analysis
* Train-test splitting
* Feature scaling
* Logistic Regression
* Model evaluation
* Cross-validation
* Hyperparameter tuning
* Model comparison

---

# 📚 Key Learning Outcomes

Through this project, I learned how to:

* Work with real-world customer data
* Perform basic data cleaning
* Detect outliers using IQR
* Perform exploratory data analysis
* Prepare features and target variables
* Split data into training and testing sets
* Apply feature scaling
* Build a classification model
* Evaluate classification performance
* Use Cross-Validation
* Perform hyperparameter tuning using GridSearchCV
* Perform hyperparameter tuning using RandomizedSearchCV
* Compare machine learning approaches

---

## 👨‍💻 Author

**Rohit Tayde**

**Skills:** Python | SQL | Pandas | NumPy | Machine Learning | Data Analysis | Data Visualization
