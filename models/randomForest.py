from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import analysis.estatisticas as stats
import analysis.targetMultiClasse as tmc

rf = RandomForestClassifier(random_state= 14)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['sqrt', 0.5, 0.7],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid = GridSearchCV(rf, param_grid, cv= 5, n_jobs= -1)
grid.fit(stats.X_train, stats.y_train)
y_pred = grid.predict(stats.X_test)

print(f"Melhores parametros: {grid.best_params_}")
print(classification_report(stats.y_test, y_pred))
