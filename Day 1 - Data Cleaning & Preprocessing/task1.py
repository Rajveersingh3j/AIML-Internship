#Necessary import =======================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import pdb



#Def function ================

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


# Task1 =========================

while True:
    try:

        #### 1.Import the dataset and explore basic info (nu ls, data types).
        #Reading CSV file
        titanic_dump_df=pd.read_csv("Titanic-Dataset.csv")

        # Display data types of each column
        print("\nData Types:\n", titanic_dump_df.dtypes)

        # Display number of missing values in each column
        print("\nMissing Values:\n", titanic_dump_df.isnull().sum())

        #### 2.Handle missing values using mean/median/imputation.
        # Handle missing values
        titanic_dump_df['Age'].fillna(titanic_dump_df['Age'].median(), inplace=True)
        titanic_dump_df['Embarked'].fillna(titanic_dump_df['Embarked'].mode()[0], inplace=True)

        # Droping Cabin column as it lots of missing value
        titanic_dump_df.drop(columns='Cabin', inplace=True)


        ####  3.Convert categorical features into numerical using encoding.
        # Encode categorical features
        titanic_dump_df_1 = titanic_dump_df.copy()

        titanic_dump_df_1['Sex'] = titanic_dump_df_1['Sex'].map({'male': 0, 'female': 1})
        titanic_dump_df_1 = pd.get_dummies(titanic_dump_df, columns=['Embarked'], drop_first=True)
        
        
        #### 4.Normalize/standardize the numerical features.
        # Standardize numerical columns
        scaler = StandardScaler()
        num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
        titanic_dump_df_1[num_cols] = scaler.fit_transform(titanic_dump_df_1[num_cols])

        #optional to remove column "Name", "Ticket"
        #df.drop(columns=['Name', 'Ticket'], inplace=True)

        #### 5.Visualize outliers using boxplots and remove them.
        # Visualize outliers
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(num_cols):
            plt.subplot(2, 2, i + 1)
            sns.boxplot(y=titanic_dump_df_1[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()


        # Apply outlier removal
        df_cleaned = remove_outliers_iqr(titanic_dump_df_1, num_cols)

        # Check new shape
        print("Original shape:", titanic_dump_df.shape)
        print("Shape after outlier removal:", df_cleaned.shape)
        # pdb.set_trace()

        break
    except Exception as e:
        print(e)
        time.sleep(1)



    
