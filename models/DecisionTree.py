from sklearn.tree import DecisionTreeClassifier
import analysis.pre_processamento as preP
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV


X = preP.df.drop(columns= ['Resultado do Teste Antigênico'])
y = preP.df['Resultado do Teste Antigênico']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 14, stratify= y)

underSampling = RandomUnderSampler(random_state= 14)
X_train, y_train = underSampling.fit_resample(X_train, y_train)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'class_weight': [None, 'balanced']
}

dtModel = DecisionTreeClassifier(random_state= 14)

grid = GridSearchCV(dtModel, param_grid, cv= 5, n_jobs= -1)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)


print(classification_report(y_test, y_pred))
