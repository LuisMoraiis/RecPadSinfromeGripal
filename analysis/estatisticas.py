import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import utils.helpers as utils

path = '/Users/luismorais/Documents/RecPadSindromeGripal/data/raw/RECPAD_SG_2020.csv'

df = pd.read_csv(path, sep= ',', low_memory= False)
prefixo_features = 'Sinais e Sintomas'

features = utils.selectFeatures(df, prefixo_features)
features.append('Vírus Detectados Antigênico')

df = df[features]
df.replace({'Sim': 1, 'Não': 0, 'Ignorado': 2}, inplace= True)
df.fillna(-1, inplace= True)
print(df.info())
print(df.sample())
print(df['Vírus Detectados Antigênico'].value_counts())

df = df.loc[df['Vírus Detectados Antigênico'].isin(['SARS-CoV-2', 'Influenza, SARS-CoV-2', 'Influenza'])]
print(df['Vírus Detectados Antigênico'].value_counts())

df['Vírus Detectados Antigênico'] = df['Vírus Detectados Antigênico'].replace(
    {'SARS-CoV-2': 10, 'Influenza, SARS-CoV-2': 30, 'Influenza': 20}
)
df = df.astype(int)

X = df.drop(columns= ['Vírus Detectados Antigênico'])
y = df['Vírus Detectados Antigênico']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 14, stratify= y)

underSampling = RandomUnderSampler(random_state= 14)
X_train, y_train = underSampling.fit_resample(X_train, y_train)
print(y_train.value_counts())
