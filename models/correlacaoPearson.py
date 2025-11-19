import pandas as pd

import analysis.pre_processamento as prep
import models.knn as knn
import models.naive_bayes as nb
import models.randomForest as rf
import models.SVM as svm
import models.xgboost as xgb

y_true = prep.y_test.reset_index(drop=True)

dic_pred_models = {
    "svm": pd.Series(svm.y_pred).reset_index(drop=True),
    "rf": pd.Series(rf.y_pred).reset_index(drop=True),
    "knn": pd.Series(knn.y_pred).reset_index(drop=True),
    "XGBoost": pd.Series(xgb.y_pred).reset_index(drop=True),
    "naive_bayes": pd.Series(nb.y_pred).reset_index(drop=True)
}

def calc_erro(dic_pred_models, y_true):
    erros = {}
    for nome, pred in dic_pred_models.items():
        erros[nome] = (pred != y_true).astype(int)
    return pd.DataFrame(erros)

def calc_corrPearson(erros_df):
    return erros_df.corr(method='pearson')
