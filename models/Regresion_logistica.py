
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
from matplotlib import style

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#load the .env file variables
load_dotenv()
connection_string = os.getenv('DATABASE_URL')
#print(connection_string)

df_raw = pd.read_csv(connection_string, sep =';')

# Paso 3 Cambio el tipo de variable segun si es atributo o no
# Crear una función que funcione en todas esas categorías es una buena práctica
for items in ['marital','education','default', 'housing', 'loan', 'contact', 'month','day_of_week', 'job']:
    df_raw[items] = pd.Categorical(df_raw[items])


df_r = df_raw.copy()

#Sugerencia para el proceso de limpieza:

#El conjunto de datos no tiene valores nulos, pero asegúrese de eliminar las filas duplicadas
df_r = df_r.drop_duplicates()

# se va a predecir el deposito a plazo fijo
# 'y' esta es la variable objetivo como se entiende que la variable de confianzza 
# puede estar correlacionada de forma positiva con la variable objetivo, ya que
# para que un cliente decida ahorrar en un depósito a plazo necesita confianza, usaremos la
# variable confianza como guia
df_r_3 = pd.DataFrame(df_r.corrwith(df_r['cons.conf.idx'], axis=0), columns=['correlacion'])
df_r_3

# estas son las variables que sacaremos del modelo por su alta correlacion
df_r_3[abs(df_r_3['correlacion'])> 0.30]

# remuevo los outliers, defino la funcion llamada "outliers"
# IQR = Q3 - Q1

def outliers(df_r, items2):
    Q1 = df_r[items2].quantile(0.25)
    Q3 = df_r[items2].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    ls = df_r.index[(df_r[items2] < lower_bound) | (df_r[items2] > upper_bound) ]

    return ls

# creo la lista de indices

index_list = []

for feature in ['age','campaign','duration','emp.var.rate','cons.price.idx']:
    index_list.extend(outliers(df_r, feature))

# defino la funcion para remover los valores

def remove(df_r, ls):
    ls = sorted(set(ls))
    df = df_r.drop(ls)
    return df

df_cleaned = remove(df_r, index_list)

df = pd.get_dummies(df_cleaned, drop_first = True) # pasa todas las variables categoricas y las pasa a dummies

df = pd.get_dummies(df, drop_first = True) # pasa todas las variables categoricas y las pasa a dummies

# Step 4:


#Time to build your model!

#Separate your target variable from the predictors
#Choose how to divide your data to evaluate the performance of your model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

X = df.drop(['y','cons.conf.idx'], axis=1)
y = df['y']

# separo el dataset de entranamiento
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=53, test_size=0.15)

# Para no incluir ningún tipo de regularización en el modelo se indica
# penalty='none'
model = LogisticRegression(solver='lbfgs', max_iter= 10000, penalty='none')

model.fit(X_train, y_train)

accuracyTrain = model.score(X_train,y_train)
accuracyTest = model.score(X_test,y_test)

print('Accuracy in Training set: ', accuracyTrain)
print('Accuracy in Testing set: ', accuracyTest)

prediccion = model.predict(X_test)

from sklearn.model_selection import  cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5).mean()
print('Accuracy in cross validation: ', scores)

prediccion = model.predict(X = X_test)
prediccion


confusion_matrix(prediccion, y_test)



