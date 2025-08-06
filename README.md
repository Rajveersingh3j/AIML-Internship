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

<strong>📘 What I Learned: </strong>

- Handling null and missing values.  
- Encoding categorical variables.  
- Feature scaling using standardization.  
- Detecting and removing outliers.  
- Building a clean dataset for machine learning.

<strong>📂 Output: </strong> A clean and preprocessed version of the Titanic dataset, ready for model training.

</details>

<details>
<summary><strong># 📊 Task 2: Exploratory Data Analysis (EDA)</strong></summary>

<strong>🎯 Objective: </strong> Understand the dataset using statistics and visualizations to uncover structure, trends, and potential issues.

<strong>🛠️ Tools Used: </strong>Pandas, Matplotlib, Seaborn, Plotly  

<strong>🗂️ Dataset: </strong> Titanic Dataset - A classic dataset used for ML and data preprocessing tasks.

<strong>✅ Steps Covered: </strong>

#### 1. Generate Summary Statistics
- Use `df.describe()` to get mean, std, min, max, and quartiles.
- Use `df.median()` and `df.mode()` for extra insight.

#### 2. Visualize Numeric Features
- Plot **histograms** to understand distributions.
- Use **boxplots** to identify outliers.

#### 3. Explore Feature Relationships
- Create a **correlation matrix** with `sns.heatmap()`.
- Use **Seaborn pairplots** for visualizing pairwise relationships.

#### 4. Identify Patterns, Trends & Anomalies
- Look for skewed distributions.
- Detect unusual values or relationships.
- Compare target (`Survived`) with features using grouped plots.

#### 5. Make Feature-Level Inferences
- Infer which features might impact the target.
- Example: Higher survival rate among females or 1st class passengers.

<strong>📘 What I Learned: </strong>

- How to perform **descriptive statistical analysis**  
- How to **visualize distributions and relationships**  
- How to identify **correlations, trends, and anomalies**  
- How to draw **basic insights** that guide feature engineering and modeling  

</details>

