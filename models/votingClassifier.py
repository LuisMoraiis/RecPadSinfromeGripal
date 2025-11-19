import correlacaoPearson as cp
import knn
import naive_bayes as nb
import randomForest as rf
import SVM
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

#import analysis.prep_dfc_completo as prepC
import analysis.pre_processamento as prep

print(cp.corrPearson)

voting = VotingClassifier(
    estimators= [
        ("svm", SVM.grid),
        ("knn", knn.grid),
        ("naiveBayes", nb.grid),
        ("xgboost", xgb.grid),
        ("rf", rf.grid)
    ],
    voting="soft"
)

modelos = {
    "SVM": SVM.grid,
    "KNN": knn.grid,
    "Naive Bayes": nb.grid,
    "XGBoost": xgb.grid,
    "Random Forest": rf.grid,
}

for nome, modelo in modelos.items():
    modelo.fit(prep.X_train, prep.y_train)
    y_pred_modelo = modelo.predict(prep.X_test)

    print(f"\n===== Classification Report: {nome} =====")
    print(classification_report(prep.y_test, y_pred_modelo))

voting.fit(prep.X_train, prep.y_train)
y_pred_voting = voting.predict(prep.X_test)

print("\n===== Classification Report: VotingClassifier =====")
print(classification_report(prep.y_test, y_pred_voting))
