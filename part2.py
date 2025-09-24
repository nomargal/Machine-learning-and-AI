# ---------- Part 2: 5-fold Cross Validation ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score

# 0) Load
df = pd.read_csv("Raisin_Dataset.csv")
X = df.drop(columns=["Class"]).to_numpy()
y = df["Class"].to_numpy()
classes = np.unique(y)
assert len(classes) == 2, "Code below assumes 2 classes."

# helpers
def mean_cov(Xc):
    mu = Xc.mean(axis=0)
    Sigma = np.cov(Xc, rowvar=False, bias=False)
    return mu, Sigma

def pooled_cov(list_Xc):
    # list_Xc: [(X_c, n_c), ...]
    d = list_Xc[0][0].shape[1]
    S = np.zeros((d, d))
    N = sum(n-1 for _, n in list_Xc)  # denom: sum(n_c-1)
    for Xc, n in list_Xc:
        _, Sc = mean_cov(Xc)
        S += (n-1) * Sc
    return S / N

def discriminant(x, mu, Sigma, prior):
    from numpy.linalg import inv, slogdet
    sign, logdet = slogdet(Sigma)
    Q = x - mu
    return -0.5*logdet - 0.5*Q @ np.linalg.inv(Sigma) @ Q + np.log(prior)

def predict_gaussian(X, params, pooled=None):
    """
    params: dict[c] -> (mu_c, Sigma_c, prior_c)  (if pooled is None)
    if pooled is not None: use same Sigma for all classes
    """
    preds = []
    for x in X:
        g = {}
        for c,(mu,Sigma,prior) in params.items():
            g[c] = discriminant(x, mu, (pooled if pooled is not None else Sigma), prior)
        preds.append(max(g, key=g.get))
    return np.array(preds)

def rates_from_cm(cm):
    # binary: rows = true [c0,c1], cols = pred [c0,c1]
    # take classes[1] as "positive" to be consistent (arbitrary but fixed)
    # map cm by true/pred indices
    # cm = [[TN, FP],
    #       [FN, TP]]  (if index 0 is negative, 1 is positive)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    TPR = TP / (TP + FN) if (TP+FN)>0 else 0.0  # recall/sensitivity
    TNR = TN / (TN + FP) if (TN+FP)>0 else 0.0  # specificity
    FPR = FP / (FP + TN) if (FP+TN)>0 else 0.0
    FNR = FN / (FN + TP) if (FN+TP)>0 else 0.0
    return {"TPR":TPR, "TNR":TNR, "FPR":FPR, "FNR":FNR}

# ensure class order fixed for confusion_matrix
cls_order = list(classes)         # e.g. ['Besni','Kecimen']
pos_cls = cls_order[1]            # treat second as "positive" in metrics

kf = KFold(n_splits=5, shuffle=True, random_state=42)

records_sep = []   # per fold metrics (separate cov)
records_pool = []  # per fold metrics (pooled cov)

fold_id = 0
for train_idx, test_idx in kf.split(X):
    fold_id += 1
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # priors from train
    priors = {c: (ytr==c).mean() for c in classes}

    # ----- Separate covariances -----
    params_sep = {}
    per_class_train = []
    for c in classes:
        Xc = Xtr[ytr==c]
        mu, Sigma = mean_cov(Xc)
        params_sep[c] = (mu, Sigma, priors[c])
        per_class_train.append((Xc, len(Xc)))

    # predict (separate)
    yhat_sep = predict_gaussian(Xte, params_sep, pooled=None)
    cm_sep = confusion_matrix(yte, yhat_sep, labels=cls_order)
    acc_sep = accuracy_score(yte, yhat_sep)
    r_sep = rates_from_cm(cm_sep)
    r_sep.update({"fold":fold_id, "accuracy":acc_sep, "mode":"separate"})
    records_sep.append(r_sep)

    # ----- Pooled covariance -----
    S_pool = pooled_cov(per_class_train)
    # reuse means/priors from params_sep, but use pooled S
    yhat_pool = predict_gaussian(Xte, params_sep, pooled=S_pool)
    cm_pool = confusion_matrix(yte, yhat_pool, labels=cls_order)
    acc_pool = accuracy_score(yte, yhat_pool)
    r_pool = rates_from_cm(cm_pool)
    r_pool.update({"fold":fold_id, "accuracy":acc_pool, "mode":"pooled"})
    records_pool.append(r_pool)

    # print per-fold summaries
    print(f"\nFold {fold_id}")
    print("Separate cov  -> Acc:", acc_sep, "\nCM:\n", cm_sep)
    print("Pooled cov    -> Acc:", acc_pool, "\nCM:\n", cm_pool)

# aggregate
df_sep  = pd.DataFrame(records_sep)
df_pool = pd.DataFrame(records_pool)
avg_sep  = df_sep.mean(numeric_only=True).to_dict()
avg_pool = df_pool.mean(numeric_only=True).to_dict()

print("\n=== Averages over 5 folds (Separate cov) ===")
print(avg_sep)
print("\n=== Averages over 5 folds (Pooled cov) ===")
print(avg_pool)

# save outputs for the report
df_sep.to_csv("part2_metrics_separate_per_fold.csv", index=False)
df_pool.to_csv("part2_metrics_pooled_per_fold.csv", index=False)

# nice summary table
summary = pd.DataFrame([
    {"mode":"separate", **avg_sep},
    {"mode":"pooled",   **avg_pool}
])
summary.to_csv("part2_metrics_summary.csv", index=False)
print("\nSaved: part2_metrics_separate_per_fold.csv, part2_metrics_pooled_per_fold.csv, part2_metrics_summary.csv")
