# Necessary Import =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import time
import pdb


# Task 4 =======================

while True:
    try:
        #### 1.Choose a binary classification dataset.
        # 1. Load the dataset
        dump_df = pd.read_csv("data.csv")
        
        """
        print("Shape of dataset:", dump_df.shape)
        print("\nColumn Names:", dump_df.columns.tolist())
        print("\nData Types:\n", dump_df.dtypes)
        print("\nMissing values:\n", dump_df.isnull().sum())
        dump_df["diagnosis"].value_counts(). # to get the unique value in a column
        """

        # 2. Drop unnecessary columns
        dump_df = dump_df.drop(columns=["id", "Unnamed: 32"]) # id column has all the unique vlaue and Unnamed has 569/569 missing value 
        
        # 3. Encode target column ('diagnosis' → 0, 1)
        label_encoder = LabelEncoder()
        dump_df['diagnosis'] = label_encoder.fit_transform(dump_df['diagnosis']) # Only diagnosis column as binary data
        # Now: 0 = Benign, 1 = Malignant





        #### 2.Train/test split and standardize features.
        # 4. Separate features (X) and target (y)
        X = dump_df.drop(columns=['diagnosis'])
        y = dump_df['diagnosis']

        # 5. Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        



        #### 3.Fit a Logistic Regression model.
        # 6. Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 7. Train Logistic Regression model
        model = LogisticRegression(max_iter=1000)  # increased iterations for convergence
        model.fit(X_train, y_train)
        



        #### 4.Evaluate with confusion matrix, precision, reca l, ROC-AUC.
        # 8. Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # probability for ROC-AUC

        # 9. Metrics
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # 10. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()




        #### 5.Tune threshold and explain sigmoid function.
        # Get predicted probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Try a new threshold (example: 0.3)
        threshold = 0.3
        y_pred_new = (y_prob >= threshold).astype(int)

        # Evaluate new threshold
        print(f"Using threshold = {threshold}")
        print("Precision:", precision_score(y_test, y_pred_new))
        print("Recall:", recall_score(y_test, y_pred_new))

        # Try multiple thresholds to see the trade-off
        thresholds = np.arange(0.1, 0.91, 0.1)
        print("\nThreshold | Precision | Recall")
        for t in thresholds:
            y_pred_t = (y_prob >= t).astype(int)
            prec = precision_score(y_test, y_pred_t)
            rec = recall_score(y_test, y_pred_t)
            print(f"{t:.1f}      | {prec:.3f}     | {rec:.3f}")

        break

    except Exception as e:
        print (e)
        time.sleep(1)
        pdb.set_trace()




