import analysis.pre_processamento as preP
import models.SVM as svm
import models.DecisionTree as dt

y_true = preP.y_test
y_pred_svm = svm.y_pred
y_pred_dt = dt.y_pred

def calc_erro(y_pred_modelo, y_true):
    erro = (y_pred_modelo != y_true).astype(int)
    return erro

def calc_corrPearson(Em1, Em2):
    return Em1.corr(Em2, method= 'pearson')
