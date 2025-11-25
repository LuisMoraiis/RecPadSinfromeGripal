import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

import analysis.pre_processamento as prep
import models.correlacaoPearson as pearson
import models.knn as knn
import models.naive_bayes as nb
import models.randomForest as rf
import models.SVM as SVM
import models.xgboost as xgb


def bias_variance_covariance_decomposition(modelos, X_train, y_train, X_test, y_test, B=50):
    n_modelos = len(modelos)
    n_test = X_test.shape[0]

    preds_boot = np.zeros((B, n_modelos, n_test))

    for b in range(B):
        Xb, yb = resample(X_train, y_train, replace=True, random_state=42 + b)

        for i, m in enumerate(modelos):
            modelo_clone = clone(m)
            modelo_clone.fit(Xb, yb)

            preds_boot[b, i] = modelo_clone.predict_proba(X_test)[:, 1]

    P_mean = preds_boot.mean(axis=0)
    P_var = preds_boot.var(axis=0)
    y_true = y_test.values.astype(float)

    bias2 = (P_mean - y_true[np.newaxis, :])**2

    variance = P_var

    covariance = np.zeros((n_modelos, n_modelos))
    for i in range(n_modelos):
        for j in range(n_modelos):
            covariance[i, j] = ((preds_boot[:, i, :] - P_mean[i])
                                * (preds_boot[:, j, :] - P_mean[j])).mean()

    noise = (y_true * (1 - y_true)).mean()

    bias2_mean = bias2.mean()
    variance_mean = variance.mean()
    covariance_mean = covariance.mean()

    decomposition = bias2_mean + variance_mean + covariance_mean + noise

    return {
        "decomposition": decomposition,
        "bias2": bias2,
        "variance": variance,
        "covariance": covariance,
        "noise": noise
    }

modelos = {
    "SVM": SVM.grid,
    "KNN": knn.grid,
    "Naive Bayes": nb.grid,
    "XGBoost": xgb.grid,
    "Random Forest": rf.grid,
}

modelos_lista = list(modelos.values())

dic_pred_mod_proba = {}
dic_pearson = {}

for nome, modelo in modelos.items():
    modelo.fit(prep.X_train, prep.y_train)
    y_pred_modelo = modelo.predict(prep.X_test)

    dic_pearson[nome] = y_pred_modelo
    dic_pred_mod_proba[nome] = modelo.predict_proba(prep.X_test)

    print(f"\n===== Classification Report: {nome} =====")
    print(classification_report(prep.y_test, y_pred_modelo))

voting = VotingClassifier(
    estimators=[
        ("svm", SVM.grid),
        ("knn", knn.grid),
        ("naiveBayes", nb.grid),
        ("xgboost", xgb.grid),
        ("rf", rf.grid)
    ],
    voting="soft"
)

voting.fit(prep.X_train, prep.y_train)

y_pred_voting = voting.predict(prep.X_test)
y_pred_voting_proba = voting.predict_proba(prep.X_test)

print("\n===== Report Voting =====")
print(classification_report(prep.y_test, y_pred_voting))

print("\n===== Correlação =====")
result_corr = pearson.calc_corrPearson(
    pearson.calc_erro(dic_pearson, prep.y_test)
)
print(result_corr)


dic_ambiguitys = {}

voting_pos = y_pred_voting_proba[:, 1]

for nome, pred_modelo in dic_pred_mod_proba.items():
    pred_pos = pred_modelo[:, 1]
    dic_ambiguitys[nome] = np.mean((pred_pos - voting_pos)**2)

print("\n===== Ambiguity =====")
for nome, amb in dic_ambiguitys.items():
    print(f"{nome}: {amb:.6f}")


print("\n===== Bias-Variance-Covariance Decomposition =====")
decomp = bias_variance_covariance_decomposition(
    modelos=modelos_lista,
    X_train=prep.X_train,
    y_train=prep.y_train,
    X_test=prep.X_test,
    y_test=prep.y_test,
    B=50
)

print("Decomposition:", decomp["decomposition"])
print("Noise:", decomp["noise"])
print("Bias² médio:", decomp["bias2"].mean())
print("Variância média:", decomp["variance"].mean())
print("Covariância média:", decomp["covariance"].mean())
