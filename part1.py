# ---------- Part 1: Discriminant Analysis ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load dataset
data = pd.read_csv("Raisin_Dataset.csv")
print("Columns:", data.columns)
print("First rows:\n", data.head())

# 1. What are the classes in this dataset? (2 pts)
classes = data["Class"].unique()
print("\nQ1. Classes in dataset:", classes)

# Split features and labels
X = data.drop(columns=["Class"])
y = data["Class"]

# 2. Calculate the log odds (scalar) and discriminant function (formula placeholder)
# log odds = ln(P(class1)/P(class2)) based on prior probabilities
p_class = data["Class"].value_counts(normalize=True)
log_odds = np.log(p_class[classes[0]] / p_class[classes[1]])
print("\nQ2. Log odds:", log_odds)
print("Discriminant function g(x) = ln(P(class1)/P(class2)) + ln(p(x|class1)/p(x|class2))")

# 3. Mean vector and covariance matrix for each class
mean_vectors = {}
cov_matrices = {}
for c in classes:
    subset = X[y == c]
    mean_vectors[c] = subset.mean().values
    cov_matrices[c] = subset.cov().values
    print(f"\nQ3. Class {c}:")
    print("Mean vector:\n", mean_vectors[c])
    print("Covariance matrix:\n", cov_matrices[c])

# 4. Generate 20 samples from each multivariate distribution
samples = {}
for c in classes:
    samples[c] = np.random.multivariate_normal(mean_vectors[c],
                                               cov_matrices[c],
                                               size=20)
    print(f"\nQ4. Generated samples for class {c}:\n", samples[c][:5])  # show first 5

# (Дальше мы продолжим шаги 5–9)
# -----------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 5. Visualize MinorAxisLength vs Perimeter distributions
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 800, 100)
y_grid = np.linspace(0, 800, 100)
X_grid, Y_grid = np.meshgrid(x, y_grid)

for c in classes:
    mean_vec = mean_vectors[c][[2, 6]]  # MinorAxisLength (index 2), Perimeter (index 6)
    cov_mat = cov_matrices[c][[2, 6]][:, [2, 6]]
    rv = multivariate_normal(mean_vec, cov_mat)
    Z = rv.pdf(np.dstack((X_grid, Y_grid)))
    ax.plot_surface(X_grid, Y_grid, Z, alpha=0.5, label=c)

ax.set_xlabel("MinorAxisLength")
ax.set_ylabel("Perimeter")
ax.set_zlabel("Density")
plt.title("Q5. Joint distribution of MinorAxisLength & Perimeter")
plt.show()
print("\nQ5. Visualized distributions. MinorAxisLength and Perimeter are not strictly univariate normal (because their joint distribution is elliptical).")

# 6. Likelihood ratio functional form
print("\nQ6. Likelihood ratio: L(x) = p(x|class1)/p(x|class2) "
      "where p(x|class) = (1/(2π)^(d/2)|Σ|^0.5) * exp(-0.5*(x-μ)^T Σ^-1 (x-μ))")

# 7. Discriminant function for each class (separate covariance)
from numpy.linalg import inv, det

def discriminant(x, mean, cov, prior):
    d = len(mean)
    return -0.5*np.log(det(cov)) - 0.5*(x-mean)@inv(cov)@(x-mean).T + np.log(prior)

priors = {c: len(X[y==c])/len(X) for c in classes}

labels_q7 = []
for i, row in X.iterrows():
    g_values = {c: discriminant(row.values, mean_vectors[c], cov_matrices[c], priors[c]) for c in classes}
    labels_q7.append(max(g_values, key=g_values.get))
labels_q7 = np.array(labels_q7)

print("\nQ7. Discriminant functions calculated. Example labels:", labels_q7[:10])

# 8. Discriminant function with pooled covariance
cov_pooled = np.zeros_like(cov_matrices[classes[0]])
for c in classes:
    cov_pooled += (len(X[y==c])-1) * cov_matrices[c]
cov_pooled /= (len(X)-len(classes))

labels_q8 = []
for i, row in X.iterrows():
    g_values = {c: discriminant(row.values, mean_vectors[c], cov_pooled, priors[c]) for c in classes}
    labels_q8.append(max(g_values, key=g_values.get))
labels_q8 = np.array(labels_q8)

print("\nQ8. Discriminant functions with pooled covariance calculated.")

# 9. Confusion matrices and accuracy
cm_q7 = confusion_matrix(y, labels_q7, labels=classes)
cm_q8 = confusion_matrix(y, labels_q8, labels=classes)
acc_q7 = accuracy_score(y, labels_q7)
acc_q8 = accuracy_score(y, labels_q8)

print("\nQ9. Confusion Matrix (separate covariance):\n", cm_q7)
print("Accuracy:", acc_q7)
print("\nQ9. Confusion Matrix (pooled covariance):\n", cm_q8)
print("Accuracy:", acc_q8)
print("Comparison: pooled vs separate covariance results differ in accuracy.")

# -----------------------------------------------------------
