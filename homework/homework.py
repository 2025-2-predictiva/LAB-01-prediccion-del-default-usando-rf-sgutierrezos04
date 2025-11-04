# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import pickle
import zipfile
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


train_data = pd.read_csv("../files/input/train_data.csv.zip", index_col=False, compression="zip")
test_data = pd.read_csv("../files/input/test_data.csv.zip", index_col=False, compression="zip")


def limpiar(df):
    df = df.rename(columns={'default payment next month': 'default'})
    df.drop('ID', axis=1, inplace=True)
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    df = df.query('MARRIAGE > 0 and EDUCATION > 0')
    df = df.dropna()
    return df


train_data = limpiar(train_data)
test_data = limpiar(test_data)

x_train = train_data.drop(columns=["default"])
y_train = train_data["default"]

x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]

colc = ['SEX', 'EDUCATION', 'MARRIAGE']


transformer = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(dtype=int), colc)
    ],
    remainder='passthrough' 
)

pipeline = Pipeline(steps=[
    ('transformer', transformer),
    ('clasi', RandomForestClassifier(n_jobs=-1, random_state=17))
])

pipeline

pipeline.fit(x_train, y_train)
print("Precisión:", pipeline.score(x_test, y_test))

param_grid = {
    'clasi__n_estimators': [180],
    'clasi__max_features': ['sqrt'],
    'clasi__min_samples_split': [10],
    'clasi__min_samples_leaf': [2],
    'clasi__bootstrap': [True],
    'clasi__max_depth': [None]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    refit=True,
    verbose=True
)
grid_search.fit(x_train, y_train)

os.makedirs('../files/models', exist_ok=True)

with gzip.open('../files/models/model.pkl.gz', 'wb') as file:
    pickle.dump(grid_search, file)

def cargar_modelo_y_predecir(data, modelo_path="../files/models/model.pkl.gz"):
    try:
        with gzip.open(modelo_path, "rb") as file:
            estimator = pickle.load(file)
        return estimator.predict(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de modelo en la ruta especificada: {modelo_path}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo o realizar predicciones: {e}")

# Uso de la función
y_train_pred = cargar_modelo_y_predecir(x_train)
y_test_pred = cargar_modelo_y_predecir(x_test)

import os
import json
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def metricas(dict_metricas):
    models_dir = '../files/output'
    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists('../files/output/metrics.json'):
        with open('../files/output/metrics.json', mode='r') as file:
            if len(file.readlines()) >= 4:
                os.remove('../files/output/metrics.json')

    with open('../files/output/metrics.json', mode='a') as file:
        file.write(str(dict_metricas).replace("'", '"') + "\n")


def evaluacion(dataset, y_true, y_pred):
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    metrics = {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "f1_score": f1
    }

    metricas(metrics)

metrics_train = evaluacion('train', y_train, y_train_pred)
metrics_test = evaluacion('test', y_test, y_test_pred)

def matriz_confusion(dataset, y_true, y_pred):
    matriz = confusion_matrix(y_true, y_pred)
    matrix_confusion = {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(matriz[0, 0]),
            "predicted_1": int(matriz[0, 1]),
        },
        "true_1": {
            "predicted_0": int(matriz[1, 0]),
            "predicted_1": int(matriz[1, 1])
        }
    }

    metricas(json.dumps(matrix_confusion))


metrics_train_cm = matriz_confusion('train', y_train, y_train_pred)
metrics_test_cm = matriz_confusion('test', y_test, y_test_pred)