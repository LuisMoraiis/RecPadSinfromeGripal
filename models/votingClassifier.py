import correlacaoPearson as cp
import knn
import naive_bayes as nb
import randomForest as rf
import SVM
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

import analysis.prep_dfc_completo as prepC

print(cp.corrPearson)

voting = VotingClassifier(
    estimators= [
        ("svm", SVM.grid),
        ("knn", knn.grid),
        ("naiveBayes", nb.grid),
        ("xgboost", xgb.grid),
        ("rf", rf.grid)
    ],
    voting= "soft"
)

voting.fit(prepC.X_train, prepC.y_train)
y_pred = voting.predict(prepC.X_test)

print(classification_report(prepC.y_test, y_pred))
