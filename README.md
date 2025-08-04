# AIML-Internship
This is my Internship Repo for AI and ML

# 🧹 Day 1: Data Cleaning & Preprocessing

**Objective:**
Learn how to clean and prepare raw data for Machine Learning models.

**Tools Used:**

* Python
* Pandas
* NumPy
* Matplotlib & Seaborn

---

## 🗂️ Dataset

Titanic Dataset – A classic dataset used for ML and data preprocessing tasks.

---

## ✅ Steps Covered

### 1. Import the Dataset & Explore Basic Information

* Load the dataset using `pandas`.
* Display the first few rows.
* Check data types and number of missing values.

### 2. Handle Missing Values

* Fill missing **Age** values using the **median**.
* Fill missing **Embarked** values using the **mode**.
* Drop **Cabin** due to excessive missing data.

### 3. Convert Categorical Features into Numerical

* Convert **Sex** using **Label Encoding** (male → 0, female → 1).
* Apply **One-Hot Encoding** to the **Embarked** column.
* Drop high-cardinality text columns like **Name** and **Ticket**.

### 4. Normalize / Standardize Numerical Features

* Standardize numerical columns (`Age`, `Fare`, `SibSp`, `Parch`) using **StandardScaler** from `sklearn`.
* This brings all numerical features to a similar scale (mean=0, std=1).

### 5. Visualize & Remove Outliers

* Use **boxplots** to visualize outliers in numerical columns.
* Remove outliers using the **IQR (Interquartile Range)** method.

---

## 📘 What I Learn

* Handling null and missing values.
* Encoding categorical variables.
* Feature scaling using standardization.
* Detecting and removing outliers.
* Building a clean dataset for machine learning.

---

## 📂 Output

A clean and preprocessed version of the Titanic dataset, ready for model training.

---

Let me know if you'd like me to add:

* Code snippets
* Dataset link
* Colab badge
* Next steps (like model training)

I'd be happy to help you extend the README.
