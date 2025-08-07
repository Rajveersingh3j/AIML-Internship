## 📊 Task 1: Data Preprocessing Interview Questions

### 1. What are the different types of missing data?

* **MCAR (Missing Completely at Random)**

  * Missingness is random and unrelated to any feature.
  * *Example:* A sensor randomly fails to record temperature.

* **MAR (Missing at Random)**

  * Missingness is related to observed data, not the missing value itself.
  * *Example:* Income might be missing more for younger individuals.

* **MNAR (Missing Not at Random)**

  * Missingness is related to the value of the missing data.
  * *Example:* People with high income may choose not to disclose it.

---

### 2. How do you handle categorical variables?

* **Label Encoding**

  * Assigns integer values to categories.
  * Best for **ordinal data**.
  * ⚠️ Risk of implying order for nominal features.

* **One-Hot Encoding**

  * Converts categories into binary columns.
  * Best for **nominal data**.
  * ⚠️ Can cause high dimensionality.

* **Frequency / Count Encoding**

  * Replaces category with its frequency/count.

* **Target Encoding**

  * Replaces category with the mean of the target variable.
  * ⚠️ Can lead to overfitting; use cross-validation or smoothing.

---

### 3. What is the difference between normalization and standardization?

| Feature               | Normalization             | Standardization                  |
| --------------------- | ------------------------- | -------------------------------- |
| Also called           | Min-Max Scaling           | Z-score Scaling                  |
| Formula               | `(x - min) / (max - min)` | `(x - mean) / std`               |
| Range                 | Scales data to \[0, 1]    | Centers data around 0 (mean = 0) |
| Sensitive to outliers | Yes                       | Less sensitive                   |
| Use case              | Non-Gaussian data         | Gaussian (normal) data           |

---

### 4. How do you detect outliers?

* **Statistical Methods**:

  * Z-Score: Values beyond ±3 std deviations
  * IQR: Outside Q1 - 1.5×IQR or Q3 + 1.5×IQR

* **Visualization**:

  * Boxplots, Histograms, Scatterplots

* **Model-Based**:

  * Isolation Forest, DBSCAN, One-Class SVM

* **Domain Knowledge**:

  * Business-specific rules

---

### 5. Why is preprocessing important in ML?

* Ensures **data quality and consistency**
* Helps **model convergence**
* Removes **noise and bias**
* Deals with **missing or imbalanced data**
* Reduces **training time**

---

### 6. What is one-hot encoding vs label encoding?

| Feature                           | One-Hot Encoding                   | Label Encoding           |
| --------------------------------- | ---------------------------------- | ------------------------ |
| Converts                          | Category → Binary columns          | Category → Numeric label |
| Suitable for                      | Nominal data                       | Ordinal data             |
| Example: ("Red", "Blue", "Green") | \[1, 0, 0], \[0, 1, 0], \[0, 0, 1] | 0, 1, 2                  |
| Risk                              | High dimensionality                | Implied order            |

---

### 7. How do you handle data imbalance?

* **Resampling**

  * Oversampling (SMOTE), Undersampling

* **Change Metrics**

  * Use F1-score, ROC-AUC, PR curve

* **Class Weights**

  * Use `class_weight='balanced'` in scikit-learn

* **Ensemble Methods**

  * Balanced Random Forest, EasyEnsemble

---

### 8. Can preprocessing affect model accuracy?

**Yes!**
Proper preprocessing can:

* Prevent algorithm errors (e.g., due to missing values)
* Improve learning and accuracy
* Ensure proper feature scaling and encoding
* Reduce noise and overfitting

---

## 📈 Task 2: Exploratory Data Analysis (EDA) Interview Questions

### 1. What is the purpose of EDA?

EDA helps in:

* Understanding data distributions
* Identifying outliers and missing values
* Spotting trends and relationships
* Preparing features for ML models

---

### 2. How do boxplots help in understanding a dataset?

Boxplots:

* Show median, quartiles, and spread
* Highlight outliers visually
* Allow comparison between groups
* Indicate skewness

---

### 3. What is correlation and why is it useful?

Correlation measures linear relationships between variables:

* Detect redundant features
* Select important features
* Understand dependencies

---

### 4. How do you detect skewness in data?

* Use `.skew()` method in Pandas
* Visual tools: Histograms, KDE plots, Boxplots
* Positive skew = right tail; Negative skew = left tail

---

### 5. What is multicollinearity?

Occurs when features are highly correlated with each other:

* Affects model coefficients
* Reduces interpretability
* Detect using correlation matrix or VIF (Variance Inflation Factor)

---

### 6. What tools do you use for EDA?

* **Python libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
* **EDA tools**: Sweetviz, Pandas Profiling, D-Tale
* **BI tools**: Excel, Power BI, Tableau

---

### 7. Can you explain a time when EDA helped you find a problem?

> "In a churn prediction project, EDA revealed a data error: tenure was incorrectly recorded in months instead of years. Boxplots exposed the anomaly. Fixing it significantly improved model performance."

---

### 8. What is the role of visualization in ML?

Visualizations help in:

* Understanding data (distribution, outliers, trends)
* Feature selection & engineering
* Model diagnostics (residuals, confusion matrix, ROC)
* Communicating insights effectively

---

## 📐 Task 3: Linear Regression – Interview Questions

### 1. **What assumptions does linear regression make?**

Linear regression makes the following key assumptions:

* **Linearity**: Relationship between independent and dependent variable is linear.
* **Independence**: Observations are independent of each other.
* **Homoscedasticity**: Constant variance of residuals.
* **Normality**: Residuals are normally distributed.
* **No multicollinearity**: Independent variables are not highly correlated.

---

### 2. **How do you interpret the coefficients?**

* Each **coefficient** represents the expected change in the dependent variable for a **one-unit increase** in the corresponding independent variable, assuming other variables are held constant.
* The **sign** (positive/negative) indicates the **direction** of the relationship.

> Example: In a model `y = 2x + 3`, the coefficient 2 means for every 1-unit increase in `x`, `y` increases by 2.

---

### 3. **What is R² score and its significance?**

* **R² (Coefficient of Determination)** measures how well the model explains the variance in the target variable.
* **Range**: 0 to 1

  * 0 → Model explains none of the variability
  * 1 → Model explains all variability

> High R² means better model fit, but beware of overfitting in complex models.

---

### 4. **When would you prefer MSE over MAE?**

* **MSE (Mean Squared Error)** penalizes **larger errors more** due to squaring.
* **MAE (Mean Absolute Error)** treats all errors equally.

**Use MSE when:**

* Large errors are more problematic.
* You want a model that is sensitive to outliers.

**Use MAE when:**

* Robustness to outliers is needed.
* Interpretability is important (error in original units).

---

### 5. **How do you detect multicollinearity?**

* **Correlation Matrix**: High correlation between independent variables.
* **Variance Inflation Factor (VIF)**:

  * VIF > 5 or 10 → Possible multicollinearity.

> Address by dropping variables, combining them, or using dimensionality reduction (e.g., PCA).

---

### 6. **What is the difference between simple and multiple regression?**

| Feature         | Simple Linear Regression | Multiple Linear Regression        |
| --------------- | ------------------------ | --------------------------------- |
| Input Variables | One independent variable | Two or more independent variables |
| Equation Form   | `y = β₀ + β₁x + ε`       | `y = β₀ + β₁x₁ + β₂x₂ + ... + ε`  |
| Use Case        | Univariate relationships | Multivariate relationships        |

---

### 7. **Can linear regression be used for classification?**

* **Not recommended** for classification tasks.
* Linear regression predicts **continuous values**, not **class labels**.
* For binary classification, use **logistic regression** instead.

> Regression used for classification can produce invalid probabilities or misclassify boundaries.

---

### 8. **What happens if you violate regression assumptions?**

* **Linearity violation**: Model may underperform or be biased.
* **Non-normal residuals**: Inaccurate confidence intervals and hypothesis testing.
* **Heteroscedasticity**: Inefficient estimates, unreliable standard errors.
* **Multicollinearity**: Unstable coefficients and high variance.
* **Autocorrelation**: Inflated R² and biased error terms in time-series data.

> Always **validate assumptions** using residual plots, Q-Q plots, or statistical tests.

---
