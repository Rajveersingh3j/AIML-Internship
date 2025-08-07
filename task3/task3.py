#Necessary import =======================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pdb



# Task 3 =======================

while True:
    try: 

        ####  1.Import and preprocess the dataset.
        # Step 1: Import the dataset
        housing_dump_df=pd.read_csv("Housing.csv")

        """ 
        #Commented the bellow line as the data was already clean and was in proper shape
        
        # Step 2: Display basic information
        print("Shape of dataset:", housing_dump_df.shape)
        print("\nColumn Names:", housing_dump_df.columns.tolist())
        print("\nData Types:\n", housing_dump_df.dtypes)

        # Step 3: Check for missing values
        print("\nMissing values:\n", housing_dump_df.isnull().sum())

        # Step 4: Drop duplicates (if any)
        df.drop_duplicates(inplace=True)
        
        # Step 5: Convert categorical variables to lowercase (optional cleanup)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower().str.strip()
        
        """

        # Step 6: Convert categorical columns using one-hot encoding (if required)
        df_encoded = pd.get_dummies(housing_dump_df, drop_first=True)
        




        ####  2.Split data into train-test sets.
        # Assuming the target variable is 'price'
        X = df_encoded.drop('price', axis=1)
        y = df_encoded['price']

        # Split the data (80% train, 20% test by default)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)




        #### 3.Fit a Linear Regression model using sklearn.linear_model.
        # Step 1: Create the model
        lr_model = LinearRegression()

        # Step 2: Fit the model to the training data
        lr_model.fit(X_train, y_train)

        # Step 3: Print model coefficients and intercept
        print("Intercept:", lr_model.intercept_)
        print("Coefficients:\n", pd.Series(lr_model.coef_, index=X_train.columns))




        #### 4.Evaluate model using MAE, MSE, R².
        # Step 1: Predict on the test set
        y_pred = lr_model.predict(X_test)

        # Step 2: Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Step 3: Print evaluation metrics
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        


        #### 5.Plot regression line and interpret coefficients.
        # Choose one feature (e.g., 'area') for simple regression plot
        feature = 'area'

        # Plot regression line for that feature
        plt.figure(figsize=(8, 6))
        sns.regplot(x=X_test[feature], y=y_test, line_kws={"color": "red"}, ci=None)
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.title("Regression Line: Area vs Price")
        plt.show()

        
        # Print all coefficients with feature names
        coefficients = pd.Series(lr_model.coef_, index=X_train.columns)
        print("Linear Regression Coefficients:\n", coefficients.sort_values(ascending=False))

        #housing_dump_df.to_csv("housing_dump_df.csv",index=False)
        #df_encoded.to_csv("housing_dump_df1.csv",index=False)

    except Exception as e:
        print(e)
        time.sleep(1)
