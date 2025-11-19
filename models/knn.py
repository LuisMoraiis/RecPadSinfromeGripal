from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import analysis.pre_processamento as prep
import analysis.prep_dfc_completo as prepC

scaler = StandardScaler()
X_train = scaler.fit_transform(prepC.X_train)
X_test = scaler.fit_transform(prepC.X_test)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(knn, param_grid, cv= 5, n_jobs= -1)
