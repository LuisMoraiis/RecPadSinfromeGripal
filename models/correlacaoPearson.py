import analysis.estatisticas as stats
import analysis.prep_dfc_completo as prepC
import analysis.targetMultiClasse as tmc
import models.knn as knn
import models.randomForest as rf
import models.SVM as svm
import models.xgboost as xgb

"""
REFATORAR O CALCULO DA CORRELAÇÃO DE PEARSON PARA O FORMATO DE MATRIX
"""

y_true = prepC.y_test

dic_pred_models = {
    "svm": svm.y_pred,
    "rf": rf.y_pred,
    "knn": knn.y_pred,
    "XGboost": xgb.y_pred
}

def calc_erro(dic_pred_models):
    erros = {}
    for key, value in dic_pred_models.items():
        erros[f"Erro {key}"] = (value != y_true).astype(int)

    return erros

def calc_corrPearson(erros):
    dic_corrPearson = {}
    for key1, value1 in erros.items():
        for key2, value2 in erros.items():
            dic_corrPearson[f"Correlação entre {key1} e {key2}:"] = value1.corr(value2, method= 'pearson')

    print(dic_corrPearson)


calc_corrPearson(calc_erro(dic_pred_models))
"""
def calc_erro(y_pred_modelo):
    erro = (y_pred_modelo != y_true).astype(int)
    return erro

def calc_corrPearson(Em1, Em2):
    return Em1.corr(Em2, method= 'pearson')
"""
