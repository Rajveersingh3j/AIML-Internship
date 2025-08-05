# AIML-Internship
This project is part of my ongoing internship in Artificial Intelligence and Machine Learning. 
<details> 
<summary><strong>🧹 Day 1: Data Cleaning & Preprocessing</strong></summary><br>

<strong>🎯 Objective: </strong>
Learn how to clean and prepare raw data for Machine Learning models.

<strong>🛠️ Tools Used: </strong> Python, Pandas, NumPy, Matplotlib & Seaborn  

<strong>🗂️ Dataset: </strong> Titanic Dataset - A classic dataset used for ML and data preprocessing tasks.

<strong>✅ Steps Covered: </strong>
#### 1. Import the Dataset & Explore Basic Information
- Load the dataset using `pandas`.
- Display the first few rows.
- Check data types and number of missing values.

#### 2. Handle Missing Values
- Fill missing **Age** values using the **median**.
- Fill missing **Embarked** values using the **mode**.
- Drop **Cabin** due to excessive missing data.

#### 3. Convert Categorical Features into Numerical
- Convert **Sex** using **Label Encoding** (male → 0, female → 1).
- Apply **One-Hot Encoding** to the **Embarked** column.
- Drop high-cardinality text columns like **Name** and **Ticket**.

#### 4. Normalize / Standardize Numerical Features
- Standardize numerical columns (`Age`, `Fare`, `SibSp`, `Parch`) using **StandardScaler** from `sklearn`.
- This brings all numerical features to a similar scale (mean=0, std=1).

#### 5. Visualize & Remove Outliers
- Use **boxplots** to visualize outliers in numerical columns.
- Remove outliers using the **IQR (Interquartile Range)** method.

<strong>📘 What I Learned</strong>

- Handling null and missing values.  
- Encoding categorical variables.  
- Feature scaling using standardization.  
- Detecting and removing outliers.  
- Building a clean dataset for machine learning.

<strong>📂 Output: </strong> A clean and preprocessed version of the Titanic dataset, ready for model training.

</details>
