from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import analysis.estatisticas as stats
import analysis.pre_processamento as preP
import analysis.targetMultiClasse as tmc

svmModel = SVC(random_state= 14)

param_grid= {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3],
    'probability': [True],
    'class_weight': [None, 'balanced']
}
grid = GridSearchCV(svmModel, param_grid, cv= 5, n_jobs= -1)


grid.fit(preP.X_train, preP.y_train)

y_pred = grid.predict(preP.X_test)

print(f"Melhores parametros: {grid.best_params_}")
print(classification_report(preP.y_test, y_pred))
