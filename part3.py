# ---------- Part 3: Regression & PCA ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# 0) Load
df = pd.read_csv("Raisin_Dataset.csv")

# features/target specification (per assignment)
features = ["Area","MajorAxisLength","MinorAxisLength","Eccentricity","Extent","ConvexArea"]
target   = "Perimeter"

X_all = df[features].to_numpy()
y_all = df[target].to_numpy()

# 1) Scatter plots: Perimeter vs each X
for col in features:
    plt.figure()
    plt.scatter(df[col], df[target], s=10)
    plt.xlabel(col)
    plt.ylabel("Perimeter")
    plt.title(f"Perimeter vs {col}")
    plt.tight_layout()
    plt.savefig(f"part3_scatter_{col}.png", dpi=150)
    # plt.show()
print("Saved 6 scatter plots: part3_scatter_*.png")

# 2) Train/Test split by first 700 samples
n_train = 700
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

# Correlation matrix on TRAIN (features + target)
corr_df = df.loc[:n_train-1, features + [target]].corr()
corr_df.to_csv("part3_train_correlations.csv", index=True)
print("\nSaved: part3_train_correlations.csv")
print("Top correlations with Perimeter (train):")
print(corr_df[target].sort_values(ascending=False))

# 3) Linear Regression on original features
lr = LinearRegression()
lr.fit(X_train, y_train)
coef_table = pd.DataFrame({"feature":features, "coef":lr.coef_}).sort_values("coef", ascending=False)
coef_table.to_csv("part3_lr_coefficients.csv", index=False)
print("\nLinear regression coefficients (descending):\n", coef_table)

# 4) Predict & MSE on test
y_pred = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)
print("\nTest MSE (linear regression):", mse_lr)

# 5) (Discussion point) Linear dependence hint: check condition number
# Large condition number -> multicollinearity risk
u, s, vh = np.linalg.svd(X_train - X_train.mean(axis=0), full_matrices=False)
cond_number = s.max() / s.min()
print("Condition number (X_train):", cond_number)

# 6) PCA on training X
pca = PCA()
pca.fit(X_train)
explained = pca.explained_variance_ratio_

# 7) Pareto chart (cumulative explained variance)
plt.figure()
plt.bar(np.arange(1, len(explained)+1), explained)
plt.plot(np.arange(1, len(explained)+1), np.cumsum(explained), marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained / Cumulative")
plt.title("PCA Variance Explained (bars) & Cumulative (line)")
plt.tight_layout()
plt.savefig("part3_pca_pareto.png", dpi=150)
# plt.show()
print("Saved: part3_pca_pareto.png")
pd.DataFrame({"PC":np.arange(1,len(explained)+1), "explained_ratio":explained}).to_csv("part3_pca_explained.csv", index=False)

# 8) Regression on first 4 PCs
pca_4 = PCA(n_components=4)
Z_train = pca_4.fit_transform(X_train)
Z_test  = pca_4.transform(X_test)

lr_pca = LinearRegression()
lr_pca.fit(Z_train, y_train)
y_pred_pca = lr_pca.predict(Z_test)
mse_pca = mean_squared_error(y_test, y_pred_pca)
print("\nTest MSE (regression on first 4 PCs):", mse_pca)

# Save predictions for report (optional)
pd.DataFrame({
    "y_test": y_test,
    "y_pred_LR": y_pred,
    "y_pred_PCA4": y_pred_pca
}).to_csv("part3_predictions.csv", index=False)
print("Saved: part3_predictions.csv")
