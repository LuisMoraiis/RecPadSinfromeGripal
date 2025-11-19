import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import utils.helpers as utils

path = '/Users/luismorais/Documents/RecPadSindromeGripal/data/raw/RECPAD_SG_2020.csv'

df = pd.read_csv(path, sep= ',', low_memory= False)
prefixo_features = 'Sinais e Sintomas'

features = utils.selectFeatures(df, prefixo_features)
features.append('Resultado do Teste Antigênico')


df = df[features]
df.replace({'Sim': 1, 'Não': 0, 'Ignorado': 2}, inplace= True)
df['Resultado do Teste Antigênico'].replace({'Negativo': 0, 'positivo': 1}, inplace= True)
df.fillna(-1, inplace= True)
df = df.astype(int)

X = df.drop(columns= ['Resultado do Teste Antigênico'])
y = df['Resultado do Teste Antigênico']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 14, stratify= y)

underSampling = RandomUnderSampler(random_state= 14)
X_train, y_train = underSampling.fit_resample(X_train, y_train)
