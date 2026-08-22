"""
Parte 2 - Ejercicio 4: Construccion de modelos de Machine Learning.

Entrena Regresion Logistica, Random Forest y Gradient Boosting (XGBoost)
para clasificar alta_cianobacteria a partir de las variables predictoras
definidas en el Ejercicio 3 (parte2/featureSelection.py), usando una
division 70/30 estratificada (para conservar la proporcion de clases,
dado el fuerte desbalance identificado en el Ejercicio 2.4).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RUTA_DATOS = "parte2/resultados/dataset_features.csv"
CARPETA_MODELOS = "parte2/resultados/modelos"
CARPETA_TABLAS = "parte2/resultados/tablas"

# Semilla y proporcion de particion fijas: se reutilizan exactamente en
# los Ejercicios 5, 6, 7, 8 y 9 para garantizar el mismo conjunto de
# prueba en todas las comparaciones (Ejercicio 4.4).
RANDOM_STATE = 42
TEST_SIZE = 0.30

PREDICTORES = [
    "ndvi", "ndwi", "B02", "B03", "B08", "B11",
    "ratio_verde_azul", "brillo_superficial", "indice_turbidez_verde_swir",
]
COLUMNA_RESPUESTA = "alta_cianobacteria"


def cargar_datos():
    return pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])


def dividir_entrenamiento_prueba(df):
    """
    Division 70/30 estratificada por la variable respuesta, para que el
    fuerte desbalance de clases (Ejercicio 2.4, ~81:1) se mantenga
    proporcionalmente tanto en entrenamiento como en prueba.
    """

    X = df[PREDICTORES]
    y = df[COLUMNA_RESPUESTA]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, idx_train, idx_test


# ============================================================
# 4.1 / 4.2 / 4.4 ENTRENAMIENTO BASE (70/30)
# ============================================================

def entrenar_modelos_base(X_train, y_train):
    """
    Entrena los tres modelos con configuraciones razonables por
    defecto. class_weight="balanced" (LR, RF) y scale_pos_weight (XGB)
    compensan el desbalance de clases documentado en el Ejercicio 2.4,
    sin necesidad de sobre/submuestrear los datos.
    """

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    modelos = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        modelo.fit(X_train, y_train)

    return modelos


def verificacion_rapida(modelos, X_test, y_test):
    """
    Chequeo rapido (accuracy y AUC) solo para confirmar que los modelos
    entrenaron correctamente. La evaluacion completa (precision,
    recall, F1, matriz de confusion, comparacion) se hace en el
    Ejercicio 5 (parte2/modelEvaluation.py).
    """

    from sklearn.metrics import accuracy_score, roc_auc_score

    print("\n--- Verificacion rapida (accuracy / ROC-AUC en prueba) ---")
    filas = []
    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        print(f"{nombre}: accuracy={acc:.4f}  roc_auc={auc:.4f}")
        filas.append({"modelo": nombre, "accuracy": acc, "roc_auc": auc})

    return pd.DataFrame(filas)


if __name__ == "__main__":
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_datos()
    X_train, X_test, y_train, y_test, idx_train, idx_test = dividir_entrenamiento_prueba(df)

    print(f"Observaciones de entrenamiento: {len(X_train)}")
    print(f"Observaciones de prueba:        {len(X_test)}")
    print(f"Tasa de positivos en train: {y_train.mean() * 100:.3f}%")
    print(f"Tasa de positivos en test:  {y_test.mean() * 100:.3f}%")

    modelos = entrenar_modelos_base(X_train, y_train)

    for nombre, modelo in modelos.items():
        joblib.dump(modelo, os.path.join(CARPETA_MODELOS, f"{nombre}_base.joblib"))

    resumen_rapido = verificacion_rapida(modelos, X_test, y_test)
    resumen_rapido.to_csv(os.path.join(CARPETA_TABLAS, "verificacion_rapida_modelos_base.csv"), index=False)
    print(f"\nModelos guardados en -> {CARPETA_MODELOS}")
