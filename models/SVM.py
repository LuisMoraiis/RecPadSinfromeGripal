from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svmModel = SVC(random_state= 14, probability= True)

param_grid= {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3],
    'probability': [True],
    'class_weight': [None, 'balanced']
}

param_grid1 = {
    'C': [0.1, 1],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}
grid = GridSearchCV(svmModel, param_grid1, cv= 5, n_jobs= -1)
