from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import analysis.pre_processamento as prep
import analysis.prep_dfc_completo as prepC

rf = RandomForestClassifier(random_state= 14)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['sqrt', 0.5, 0.7],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_grid1 = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt'],
    'max_depth': [30, 40],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [4, 6],
    'bootstrap': [True, False]
}

grid = GridSearchCV(rf, param_grid1, cv= 5, n_jobs= -1)
