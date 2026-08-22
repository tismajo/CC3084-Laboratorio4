"""
Parte 2 - Ejercicio 5: Evaluacion de los modelos.

Evalua los tres modelos finales (ajustados en el Ejercicio 4) sobre el
mismo conjunto de prueba (70/30 estratificado, semilla 42) para poder
comparar de forma justa.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from models import cargar_datos, dividir_entrenamiento_prueba, CARPETA_MODELOS, CARPETA_TABLAS

NOMBRES_MODELOS = ["logistic_regression", "random_forest", "xgboost"]


def cargar_modelos_finales():
    modelos = {}
    for nombre in NOMBRES_MODELOS:
        ruta = os.path.join(CARPETA_MODELOS, f"{nombre}_final.joblib")
        modelos[nombre] = joblib.load(ruta)
    return modelos


# ============================================================
# 5.1 METRICAS Y MATRIZ DE CONFUSION POR MODELO
# ============================================================

def evaluar_modelo(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    matriz = confusion_matrix(y_test, y_pred)

    print(f"\n--- {nombre} ---")
    for k, v in metricas.items():
        if k != "modelo":
            print(f"{k}: {v:.4f}")
    print("Matriz de confusion [[TN, FP], [FN, TP]]:")
    print(matriz)

    return metricas, matriz


def evaluar_todos(modelos, X_test, y_test):
    filas = []
    matrices = {}
    for nombre, modelo in modelos.items():
        metricas, matriz = evaluar_modelo(nombre, modelo, X_test, y_test)
        filas.append(metricas)
        matrices[nombre] = matriz
    return pd.DataFrame(filas), matrices


# ============================================================
# 5.2 COMPARACION DE MODELOS
# ============================================================

def comparar_modelos(tabla_metricas):
    print("\n--- Comparacion de modelos (conjunto de prueba) ---")
    print(tabla_metricas.to_string(index=False))

    mejor_por_f1 = tabla_metricas.loc[tabla_metricas["f1"].idxmax(), "modelo"]
    mejor_por_auc = tabla_metricas.loc[tabla_metricas["roc_auc"].idxmax(), "modelo"]

    print(f"\nMejor modelo por F1: {mejor_por_f1}")
    print(f"Mejor modelo por ROC-AUC: {mejor_por_auc}")

    return mejor_por_f1, mejor_por_auc


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_datos()
    _, X_test, _, y_test, _, idx_test = dividir_entrenamiento_prueba(df)

    modelos = cargar_modelos_finales()
    tabla_metricas, matrices = evaluar_todos(modelos, X_test, y_test)

    tabla_metricas.to_csv(os.path.join(CARPETA_TABLAS, "comparacion_modelos_metricas.csv"), index=False)

    for nombre, matriz in matrices.items():
        pd.DataFrame(
            matriz, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]
        ).to_csv(os.path.join(CARPETA_TABLAS, f"matriz_confusion_{nombre}.csv"))

    comparar_modelos(tabla_metricas)
