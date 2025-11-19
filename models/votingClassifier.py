from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report

#import analysis.prep_dfc_completo as prepC
import analysis.pre_processamento as prep
import models.correlacaoPearson as pearson
import models.knn as knn
import models.naive_bayes as nb
import models.randomForest as rf
import models.SVM as SVM
import models.xgboost as xgb

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

dic_pearson= {}
for nome, modelo in modelos.items():
    modelo.fit(prep.X_train, prep.y_train)
    y_pred_modelo = modelo.predict(prep.X_test)

    dic_pearson[nome] = y_pred_modelo

    print(f"\n===== Classification Report: {nome} =====")
    print(classification_report(prep.y_test, y_pred_modelo))

voting.fit(prep.X_train, prep.y_train)
y_pred_voting = voting.predict(prep.X_test)

print("\n===== Correlação =====")
print(pearson.calc_corrPearson(pearson.calc_erro(dic_pearson, prep.y_test)))

print("\n===== Classification Report: VotingClassifier =====")
print(classification_report(prep.y_test, y_pred_voting))
