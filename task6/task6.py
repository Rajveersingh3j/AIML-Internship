import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Optional: ace_tools for nicer DataFrame display in some environments
try:
    import ace_tools
    ACE_TOOLS_AVAILABLE = True
except Exception:
    ACE_TOOLS_AVAILABLE = False

# Load dataset
df = pd.read_csv("Iris.csv")

# Drop ID column if present
id_cols = [c for c in df.columns if c.lower() == "id"]
if id_cols:
    df = df.drop(columns=id_cols)

# Detect target column
possible_targets = [c for c in df.columns if c.lower() in ("species", "class", "target", "label")]
target_col = possible_targets[0] if possible_targets else df.columns[-1]

# Encode target labels
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Features and target
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# K values to try
k_values = [1, 3, 5, 7, 9]
summary = []

# Fit PCA (2D) on training data for visualization
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_pca_all = np.vstack([X_train_pca, X_test_pca])

print("Classes (label -> name):", {int(i): name for i, name in enumerate(le.classes_)})
print("-" * 60)

for k in k_values:
    # Train & evaluate on full standardized features
    knn_full = KNeighborsClassifier(n_neighbors=k)
    knn_full.fit(X_train, y_train)
    y_pred_full = knn_full.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred_full)
    cm_full = confusion_matrix(y_test, y_pred_full)

    # Train & evaluate on 2D PCA features (for plotting)
    knn_pca = KNeighborsClassifier(n_neighbors=k)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    cm_pca = confusion_matrix(y_test, y_pred_pca)

    summary.append({"k": k, "accuracy_full": acc_full, "cm_full": cm_full, "accuracy_pca": acc_pca, "cm_pca": cm_pca})

    # Print results
    print(f"k = {k:>2} | accuracy (full features) = {acc_full:.4f} | accuracy (PCA2) = {acc_pca:.4f}")
    print("Confusion matrix (full features):")
    print(cm_full)
    print("Confusion matrix (PCA 2D):")
    print(cm_pca)
    print("-" * 60)

# Summary table
summary_df = pd.DataFrame([{"k": s["k"], "accuracy_full": s["accuracy_full"], "accuracy_pca": s["accuracy_pca"]} for s in summary])
if ACE_TOOLS_AVAILABLE:
    ace_tools.display_dataframe_to_user("KNN summary (accuracy)", summary_df)
else:
    print("\nSummary table:")
    print(summary_df.to_string(index=False))

# Show confusion matrices per k
for s in summary:
    k = s["k"]
    cmf = pd.DataFrame(s["cm_full"], index=le.classes_, columns=le.classes_)
    cmp = pd.DataFrame(s["cm_pca"], index=le.classes_, columns=le.classes_)
    title_full = f"Confusion matrix (full features) k={k}"
    title_pca = f"Confusion matrix (PCA2) k={k}"
    if ACE_TOOLS_AVAILABLE:
        ace_tools.display_dataframe_to_user(title_full, cmf)
        ace_tools.display_dataframe_to_user(title_pca, cmp)
    else:
        print(title_full)
        print(cmf)
        print(title_pca)
        print(cmp)

# Decision boundary plots (one figure per k) in PCA 2D space
x_min, x_max = X_pca_all[:, 0].min() - 1.0, X_pca_all[:, 0].max() + 1.0
y_min, y_max = X_pca_all[:, 1].min() - 1.0, X_pca_all[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = np.c_[xx.ravel(), yy.ravel()]

for s in summary:
    k = s["k"]
    knn_pca = KNeighborsClassifier(n_neighbors=k)
    knn_pca.fit(X_train_pca, y_train)
    Z = knn_pca.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)  # default colormap used
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=40, marker='o', label='train')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, s=70, marker='x', label='test')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"KNN decision boundary (k={k}) — PCA 2D projection")
    mapping_text = "; ".join([f"{i}={name}" for i, name in enumerate(le.classes_)])
    plt.text(0.99, 0.01, mapping_text, transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=9)
    plt.show()
