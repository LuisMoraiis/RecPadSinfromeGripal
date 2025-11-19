import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

naiveBayes = GaussianNB()
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=10)
}
grid = GridSearchCV(naiveBayes, param_grid, cv= 5, n_jobs= -1)
