#Necessary import =======================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pdb



#Task 2 ================


while True: 

    try:
        #### 1.Generate summary statistics (mean, median, std, etc.).
        # Summary statistics for numeric columns (mean, std, min, max, etc.)
        #Reading CSV file
        titanic_dump_df=pd.read_csv("Titanic-Dataset.csv")

        numeric_summary = titanic_dump_df.describe()

        # Summary statistics for all columns (including object/categorical)
        full_summary = titanic_dump_df.describe(include='all')

        # Median for numeric columns (not included in .describe())
        medians = titanic_dump_df.median(numeric_only=True)

        # Print results
        print("📊 Numeric Summary Statistics:\n", numeric_summary)
        print("\n📊 Median Values:\n", medians)
        print("\n📊 Full Summary (Including Categorical):\n", full_summary)


        #### 2.Create histograms and boxplots for numeric features.
        # Select numeric columns
        numeric_cols = titanic_dump_df.select_dtypes(include='number').columns

        # Set up the plotting grid size
        n_cols = 2  # histogram and boxplot per variable
        n_rows = len(numeric_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        # Plot histograms and boxplots for each numeric column
        for i, col in enumerate(numeric_cols):
            # Histogram
            axes[i, 0].hist(titanic_dump_df[col].dropna(), bins=30, edgecolor='black')
            axes[i, 0].set_title(f'Histogram of {col}')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel('Frequency')

            # Boxplot
            axes[i, 1].boxplot(titanic_dump_df[col].dropna(), vert=False)
            axes[i, 1].set_title(f'Boxplot of {col}')
            axes[i, 1].set_xlabel(col)

        plt.tight_layout()
        plt.show()
        

        #### 3.Use pairplot/correlation matrix for feature relationships.
        # Select numeric features
        numeric_df = titanic_dump_df.select_dtypes(include='number')

        # Correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

        # Pairplot
        sns.pairplot(numeric_df)
        plt.suptitle("Pairplot of Numeric Features", y=1.02)
        plt.show()


        #### 4.Identify patterns, trends, or anomalies in the data.  in python.
        # Set up the figure
        plt.figure(figsize=(18, 18))

        # Plot 1: Survival by Sex
        plt.subplot(3, 2, 1)
        sns.countplot(data=titanic_dump_df, x='Sex', hue='Survived')
        plt.title('Survival by Sex')
        plt.xlabel('Sex')
        plt.ylabel('Count')

        # Plot 2: Survival by Pclass
        plt.subplot(3, 2, 2)
        sns.countplot(data=titanic_dump_df, x='Pclass', hue='Survived')
        plt.title('Survival by Passenger Class')
        plt.xlabel('Pclass')
        plt.ylabel('Count')

        # Plot 3: Age Distribution by Survival
        plt.subplot(3, 2, 3)
        sns.kdeplot(data=titanic_dump_df[titanic_dump_df['Survived'] == 1]['Age'].dropna(), label='Survived', fill=True)
        sns.kdeplot(data=titanic_dump_df[titanic_dump_df['Survived'] == 0]['Age'].dropna(), label='Not Survived', fill=True)
        plt.title('Age Distribution by Survival')
        plt.xlabel('Age')
        plt.legend()

        # Plot 4: Fare Distribution by Survival
        plt.subplot(3, 2, 4)
        sns.boxplot(data=titanic_dump_df, x='Survived', y='Fare')
        plt.title('Fare Distribution by Survival')
        plt.xlabel('Survived')
        plt.ylabel('Fare')

        # Plot 5: Survival Rate by SibSp
        plt.subplot(3, 2, 5)
        sns.barplot(data=titanic_dump_df, x='SibSp', y='Survived')
        plt.title('Survival Rate by SibSp')
        plt.xlabel('Number of Siblings/Spouses')
        plt.ylabel('Survival Rate')

        # Plot 6: Missing Data Heatmap
        plt.subplot(3, 2, 6)
        sns.heatmap(titanic_dump_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Missing Values Heatmap")

        plt.tight_layout()
        plt.show()
        break

    except Exception as e:
        print(e)
        time.sleep(1)
