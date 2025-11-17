from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import analysis.estatisticas as stats
import analysis.targetMultiClasse as tmc

scaler = StandardScaler()
X_train = scaler.fit_transform(stats.X_train)
X_test = scaler.fit_transform(stats.X_test)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(knn, param_grid, cv= 5, n_jobs= -1)

grid.fit(X_train, stats.y_train)
y_pred = grid.predict(X_test)

print(f"Melhores parametros: {stats.best_params_}")
print(classification_report(stats.y_test, y_pred))
