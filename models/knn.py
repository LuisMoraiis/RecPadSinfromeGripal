from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import analysis.targetMultiClasse as tmc

scaler = StandardScaler()
X_train = scaler.fit_transform(tmc.X_train)
X_test = scaler.fit_transform(tmc.X_test)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(knn, param_grid, cv= 5, n_jobs= -1)

grid.fit(X_train, tmc.y_train)
y_pred = grid.predict(X_test)

print(f"Melhores parametros: {grid.best_params_}")
print(classification_report(tmc.y_test, y_pred))
