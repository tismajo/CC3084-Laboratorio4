"""
Parte 2 - Ejercicio 7: Generalizacion entre lagos.

Evalua si un modelo entrenado con observaciones de un lago es capaz de
generalizar al otro lago (Experimentos A y B).

Se reutilizan los mismos hiperparametros seleccionados en el
Ejercicio 4.3 para los tres modelos, de modo que la unica diferencia
entre escenarios sea de donde provienen los datos de entrenamiento y
prueba (no un ajuste de hiperparametros distinto por escenario).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RUTA_DATOS = "parte2/resultados/dataset_features.csv"
CARPETA_TABLAS = "parte2/resultados/tablas"

RANDOM_STATE = 42
TEST_SIZE = 0.30

PREDICTORES = [
    "ndvi", "ndwi", "B02", "B03", "B08", "B11",
    "ratio_verde_azul", "brillo_superficial", "indice_turbidez_verde_swir",
]
COLUMNA_RESPUESTA = "alta_cianobacteria"


def cargar_datos():
    return pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])


def construir_modelos(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    return {
        "logistic_regression": LogisticRegression(C=100, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def entrenar_y_evaluar(X_train, y_train, X_test, y_test, escenario):
    modelos = construir_modelos(y_train)
    filas = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        filas.append({
            "escenario": escenario,
            "modelo": nombre,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "positivos_train_pct": round(y_train.mean() * 100, 4),
            "positivos_test_pct": round(y_test.mean() * 100, 4),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float("nan"),
        })
        print(f"[{escenario}] {nombre}: f1={filas[-1]['f1']:.4f} recall={filas[-1]['recall']:.4f} roc_auc={filas[-1]['roc_auc']:.4f}")

    return pd.DataFrame(filas)


# ============================================================
# 7.1 / 7.2 / 7.3 EXPERIMENTOS DE GENERALIZACION ENTRE LAGOS
# ============================================================

def experimentos_entre_lagos(df):
    atitlan = df[df["lago"] == "Atitlan"]
    amatitlan = df[df["lago"] == "Amatitlan"]

    resultados = []

    # Experimento A: entrenar con Atitlan, evaluar con Amatitlan
    X_train_a, y_train_a = atitlan[PREDICTORES], atitlan[COLUMNA_RESPUESTA]
    X_test_a, y_test_a = amatitlan[PREDICTORES], amatitlan[COLUMNA_RESPUESTA]
    resultados.append(entrenar_y_evaluar(X_train_a, y_train_a, X_test_a, y_test_a, "A_train_Atitlan_test_Amatitlan"))

    # Experimento B: entrenar con Amatitlan, evaluar con Atitlan
    X_train_b, y_train_b = amatitlan[PREDICTORES], amatitlan[COLUMNA_RESPUESTA]
    X_test_b, y_test_b = atitlan[PREDICTORES], atitlan[COLUMNA_RESPUESTA]
    resultados.append(entrenar_y_evaluar(X_train_b, y_train_b, X_test_b, y_test_b, "B_train_Amatitlan_test_Atitlan"))

    return pd.concat(resultados, ignore_index=True)


# ============================================================
# 7.3 (cont.) / 7.4 REFERENCIA: MISMO LAGO EN ENTRENAMIENTO Y PRUEBA
# ============================================================

def experimentos_mismo_lago(df):
    resultados = []
    for nombre_lago in ["Atitlan", "Amatitlan"]:
        sub = df[df["lago"] == nombre_lago]
        X = sub[PREDICTORES]
        y = sub[COLUMNA_RESPUESTA]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        resultados.append(entrenar_y_evaluar(X_train, y_train, X_test, y_test, f"mismo_lago_{nombre_lago}"))

    return pd.concat(resultados, ignore_index=True)


# ============================================================
# 7.5 / 7.6 COMPARACION Y DISCUSION
# ============================================================

DISCUSION_GENERALIZACION = """
--- Discusion: generalizacion entre lagos (Ejercicio 7.5-7.6) ---

7.5 Un modelo entrenado en un lago NO generaliza adecuadamente al
otro. En ambas direcciones el desempeno cae drasticamente frente al
escenario de referencia (mismo lago): por ejemplo, Random Forest pasa
de recall=0.454 (mismo lago, Atitlan) a recall=0.001 al evaluarse en
Amatitlan tras entrenarse solo en Atitlan; XGBoost pasa de recall=0.934
(mismo lago, Amatitlan) a recall=0.122-0.189 en los experimentos
cruzados. Regresion Logistica es la excepcion parcial: mantiene recall
alto entre lagos, pero unicamente porque predice positivo con mucha
mas frecuencia (precision tan baja como 0.057-0.137), no porque
capture un patron transferible.

7.6 Posibles causas de esta falta de generalizacion:

  - Linea base ecologica muy distinta: Amatitlan es un lago somero,
    hipereutrofico, con floraciones frecuentes e intensas (positivos
    en ~11% de sus pixeles); Atitlan es un lago profundo y
    relativamente oligotrofico, con floraciones raras y localizadas
    (positivos en ~0.06% de sus pixeles). Un modelo entrenado en un
    regimen ecologico no ha visto ejemplos suficientes del otro
    regimen.

  - Rango de valores espectrales distinto: la turbidez, el color del
    agua y la reflectancia base difieren entre ambos lagos incluso en
    ausencia de cianobacteria (diferente geologia, profundidad,
    entorno urbano/agricola circundante), por lo que un umbral de
    decision aprendido en un lago puede no ser valido en el otro
    (covariate shift).

  - Desbalance extremo en Atitlan: con solo 322 pixeles positivos en
    toda la serie, un modelo entrenado alli tiene muy poca senal para
    aprender un patron generalizable, mientras que un modelo entrenado
    en Amatitlan aprende un patron ajustado a su propio regimen de alta
    frecuencia de floraciones, que no aplica al regimen raro de
    Atitlan.

Conclusion practica: para un sistema operativo de monitoreo, esto
sugiere que se necesita un modelo (o al menos un recalibrado de
umbral/hiperparametros) especifico por lago, en vez de un unico modelo
entrenado en un lago y aplicado directamente al otro.
"""


def imprimir_discusion():
    print(DISCUSION_GENERALIZACION)


if __name__ == "__main__":
    os.makedirs(CARPETA_TABLAS, exist_ok=True)

    df = cargar_datos()

    print("=== Experimentos de generalizacion entre lagos ===")
    tabla_entre_lagos = experimentos_entre_lagos(df)

    print("\n=== Experimentos de referencia: mismo lago (entrenamiento y prueba) ===")
    tabla_mismo_lago = experimentos_mismo_lago(df)

    tabla_completa = pd.concat([tabla_entre_lagos, tabla_mismo_lago], ignore_index=True)
    tabla_completa.to_csv(os.path.join(CARPETA_TABLAS, "generalizacion_entre_lagos.csv"), index=False)

    print("\n--- Tabla completa de comparacion ---")
    print(tabla_completa.to_string(index=False))

    imprimir_discusion()
