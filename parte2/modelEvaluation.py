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
    fbeta_score,
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
        "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
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


# ============================================================
# 5.3 ANALISIS AMBIENTAL DE LOS ERRORES
# ============================================================

ANALISIS_AMBIENTAL = """
--- Analisis ambiental de los errores de clasificacion (Ejercicio 5.3) ---

Falso positivo (predecir alta presencia cuando NO la hay):
  Consecuencia: cierre preventivo o alerta innecesaria de una zona del
  lago, con costo economico/turistico y posible perdida de confianza en
  el sistema de alerta si se repite con frecuencia. Es un costo
  principalmente operativo y reversible.

Falso negativo (no detectar una zona que SI tiene alta presencia):
  Consecuencia: exposicion de banistas/turistas/pescadores a agua con
  cianobacterias potencialmente toxicas sin advertencia, con riesgo de
  efectos agudos a la salud (irritacion de piel y mucosas, problemas
  gastrointestinales, y en exposiciones severas efectos hepatotoxicos o
  neurotoxicos segun la especie de cianobacteria). Es un costo sobre
  salud publica, potencialmente irreversible.

Conclusion: en este problema, un falso negativo es claramente mas
grave que un falso positivo, por lo que interesa priorizar el RECALL de
la clase alta_cianobacteria (minimizar los casos de alta presencia que
el modelo deja pasar), sin descuidar por completo la precision (una
tasa de falsos positivos demasiado alta tambien erosiona la utilidad
practica del sistema). Por eso se usa F2-score (fbeta con beta=2, que
pondera el recall el doble que la precision) como la metrica mas
adecuada para comparar los modelos en este contexto ambiental, en vez
de F1 o accuracy.
"""


def imprimir_analisis_ambiental():
    print(ANALISIS_AMBIENTAL)


def seleccionar_mejor_modelo_ambiental(tabla_metricas):
    mejor = tabla_metricas.loc[tabla_metricas["f2"].idxmax(), "modelo"]
    print(f"\nMejor modelo segun F2 (prioriza recall, criterio ambiental): {mejor}")
    return mejor


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

    imprimir_analisis_ambiental()
    mejor_modelo_ambiental = seleccionar_mejor_modelo_ambiental(tabla_metricas)

    with open(os.path.join(CARPETA_TABLAS, "mejor_modelo_ambiental.txt"), "w", encoding="utf-8") as f:
        f.write(mejor_modelo_ambiental)
