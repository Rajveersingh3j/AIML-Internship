import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

# --------- User parameters ---------
file_path = "heart.csv"             # change if your file is elsewhere
random_state = 42
test_size = 0.20
cv_folds = 5
max_depth_search = range(1, 21)               # depths to check for overfitting analysis
rf_n_estimators = 200
# -----------------------------------

# Load data
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV file not found: {file_path}")
df = pd.read_csv(file_path)

# --- Auto-detect binary target column ---
target_col = None
if "target" in df.columns:
    target_col = "target"
elif "Target" in df.columns:
    target_col = "Target"
else:
    # find columns with exactly 2 unique values (likely binary)
    bin_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 2]
    if len(bin_cols) >= 1:
        target_col = bin_cols[0]
    else:
        # fallback to last column
        target_col = df.columns[-1]

print("Detected target column:", target_col)

# Split X/y
y = df[target_col]
X = df.drop(columns=[target_col])

# --- Preprocessing: numeric vs categorical ---
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Treat small-integer numeric columns with low cardinality as categorical
for c in list(num_cols):
    if X[c].nunique() <= 10:
        # move to categorical
        num_cols.remove(c)
        if c not in cat_cols:
            cat_cols.append(c)

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ],
    remainder="drop"
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

# Pipelines
dt_pipeline = Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(random_state=random_state))])
rf_pipeline = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1))])

# -------------------------
# 1) Train Decision Tree & visualize top levels
# -------------------------
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)
dt_test_acc = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree (default) test accuracy: {dt_test_acc:.4f}")
print("Decision Tree classification report:\n", classification_report(y_test, y_pred_dt))

# Build feature names after fitting preprocessor to training data
preprocessor_fit = preprocessor.fit(X_train)
feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    ohe = preprocessor_fit.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

# Visualize tree (top 3 levels) - keep it readable
clf_dt = dt_pipeline.named_steps["clf"]
plt.figure(figsize=(18, 10))
plot_tree(clf_dt, feature_names=feature_names, class_names=[str(c) for c in np.unique(y)], max_depth=3, filled=True, fontsize=8)
plt.title("Decision Tree (top 3 levels)")
plt.tight_layout()
plt.show()

# -------------------------
# 2) Analyze overfitting: sweep max_depth and plot train vs CV accuracy
# -------------------------
train_scores = []
cv_scores = []
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

for d in max_depth_search:
    model = Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(max_depth=d, random_state=random_state))])
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    cv_scores.append(np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(list(max_depth_search), train_scores, marker='o', label="Train Accuracy")
plt.plot(list(max_depth_search), cv_scores, marker='o', label=f"CV ({cv_folds}-fold) Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Train vs CV accuracy by max_depth")
plt.legend()
plt.grid(True)
plt.show()

# Choose best depth by CV (first maximum)
best_idx = int(np.argmax(cv_scores))
best_depth = list(max_depth_search)[best_idx]
print(f"Best max_depth by CV: {best_depth} (CV accuracy={cv_scores[best_idx]:.4f})")

# Retrain best decision tree
best_dt_pipeline = Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(max_depth=best_depth, random_state=random_state))])
best_dt_pipeline.fit(X_train, y_train)
y_pred_best_dt = best_dt_pipeline.predict(X_test)
best_dt_acc = accuracy_score(y_test, y_pred_best_dt)
print(f"Decision Tree (max_depth={best_depth}) test accuracy: {best_dt_acc:.4f}")
print("Classification report (best DT):\n", classification_report(y_test, y_pred_best_dt))

# -------------------------
# 3) Train Random Forest and compare accuracy
# -------------------------
rf_pipeline.set_params(clf__n_estimators=rf_n_estimators, clf__max_depth=None)
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest (n_estimators={rf_n_estimators}) test accuracy: {rf_test_acc:.4f}")
print("Random Forest classification report:\n", classification_report(y_test, y_pred_rf))

# -------------------------
# 4) Interpret feature importances (from Random Forest)
# -------------------------
rf_clf = rf_pipeline.named_steps["clf"]
pre_fit = rf_pipeline.named_steps["pre"].fit(X_train)  # ensure transformers fitted
feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    ohe = pre_fit.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))

importances = rf_clf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 15 feature importances (Random Forest):\n", feat_imp.head(15))

# Plot top importances
plt.figure(figsize=(10, 6))
feat_imp.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# -------------------------
# 5) Evaluate using cross-validation for tuned DT and RF
# -------------------------
dt_cv_scores = cross_val_score(best_dt_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
rf_cv_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"\nDecision Tree (max_depth={best_depth}) CV accuracy: mean={dt_cv_scores.mean():.4f}, std={dt_cv_scores.std():.4f}")
print(f"Random Forest CV accuracy: mean={rf_cv_scores.mean():.4f}, std={rf_cv_scores.std():.4f}")

# Confusion matrices on the test set
print("\nConfusion matrix (Decision Tree best) - test set:")
print(confusion_matrix(y_test, y_pred_best_dt))

print("\nConfusion matrix (Random Forest) - test set:")
print(confusion_matrix(y_test, y_pred_rf))

# Summary table
summary = pd.DataFrame({
    "model": ["DecisionTree_default", f"DecisionTree_depth_{best_depth}", "RandomForest"],
    "test_accuracy": [dt_test_acc, best_dt_acc, rf_test_acc],
    "cv_mean": [
        np.mean(cross_val_score(dt_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)),
        dt_cv_scores.mean(),
        rf_cv_scores.mean()
    ],
    "cv_std": [
        np.std(cross_val_score(dt_pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)),
        dt_cv_scores.std(),
        rf_cv_scores.std()
    ]
})
print("\nModel summary:\n", summary)

# Optionally, save the plots and summary
# plt.savefig("tree_plot.png"); plt.savefig("importance_plot.png")
# summary.to_csv("model_summary.csv", index=False)
