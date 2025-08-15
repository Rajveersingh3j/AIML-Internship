import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 1. Load and prepare dataset
# ================================
df = pd.read_csv("breast-cancer.csv")

# Separate features and target
X = df.drop(columns=['id', 'diagnosis'])
y = LabelEncoder().fit_transform(df['diagnosis'])  # M=1, B=0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

# ================================
# 2. Train SVM (Linear and RBF)
# ================================
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
svm_rbf.fit(X_train, y_train)

# ================================
# 3. Visualize Decision Boundaries
# ================================
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

plot_decision_boundary(svm_linear, X_train, y_train, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, X_train, y_train, "SVM with RBF Kernel")

# ================================
# 4. Hyperparameter tuning (RBF)
# ================================
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.5, 1]
}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# ================================
# 5. Cross-validation evaluation
# ================================
best_params = grid_search.best_params_
best_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
scores = cross_val_score(best_model, X_train, y_train, cv=5)

print("Best RBF parameters:", best_params)
print("Mean CV accuracy:", scores.mean())
