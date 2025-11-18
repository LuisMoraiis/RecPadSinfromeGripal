from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import analysis.pre_processamento as prep
import analysis.prep_dfc_completo as prepC

xgboost = GradientBoostingClassifier(random_state= 14)

param_grid = {
    'n_estimators': [100, 200, 500, 800],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'gamma': [0, 0.1, 0.5],
    'lambda': [0.1, 1, 10],
    'alpha': [0.1, 1, 10],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'subsample': [0.7, 0.9, 1.0]
}

grid = GridSearchCV(xgboost, param_grid, cv= 5, n_jobs= -1)

grid.fit(prepC.X_train, prepC.y_train)
y_pred = grid.predict(prepC.X_test)

print("XGboost:")
print(grid.best_params_)
print(classification_report(prepC.y_test, y_pred))
