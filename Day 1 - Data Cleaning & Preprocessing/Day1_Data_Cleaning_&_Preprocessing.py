import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Reading CSV file
titanic_dataset_dump_df=pd.read_csv("Titanic-Dataset.csv")

###1.Import the dataset and explore basic info (nulls, data types).
# Display first few rows
print(titanic_dataset_dump_df.head())

# Display data types of each column
print("\nData Types:\n", titanic_dataset_dump_df.dtypes)

# Display number of missing values in each column
print("\nMissing Values:\n", titanic_dataset_dump_df.isnull().sum())



###2.Handle missing values using mean/median/imputation.
# 1. Fill missing 'Age' values with the median
titanic_dataset_dump_df['Age'].fillna(titanic_dataset_dump_df['Age'].median(), inplace=True)

# 2. Fill missing 'Embarked' values with the mode (most frequent value)
titanic_dataset_dump_df['Embarked'].fillna(titanic_dataset_dump_df['Embarked'].mode()[0], inplace=True)

# 3. Drop 'Cabin' column due to too many missing values (optional)
titanic_dataset_dump_df.drop(columns='Cabin', inplace=True)

# Check for remaining missing values
print("Missing values after imputation:\n", titanic_dataset_dump_df.isnull().sum())




###3.Convert categorical features into numerical using encoding.
# Convert 'Sex' column using label encoding (binary category)
titanic_dataset_dump_df['Sex'] = titanic_dataset_dump_df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' column using one-hot encoding
titanic_dataset_dump_df = pd.get_dummies(titanic_dataset_dump_df, columns=['Embarked'], drop_first=True)

# Display result
print(titanic_dataset_dump_df.head())





###4.Normalize/standardize the numerical features.
# Identify numerical columns to scale (excluding target 'Survived')
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Initialize scaler and apply
scaler = StandardScaler()
titanic_dataset_dump_df[num_cols] = scaler.fit_transform(titanic_dataset_dump_df[num_cols])

# Show result
print(titanic_dataset_dump_df.head())


###5.Visualize outliers using boxplots and remove them.
# Visualize outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(y=titanic_dataset_dump_df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Function to remove outliers using IQR
def remove_outliers_iqr(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

# Apply outlier removal
df_cleaned = remove_outliers_iqr(titanic_dataset_dump_df, num_cols)

# Check new shape
print("Original shape:", titanic_dataset_dump_df.shape)
print("Shape after outlier removal:", df_cleaned.shape)
