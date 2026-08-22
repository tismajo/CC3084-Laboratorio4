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


# ============================================================
# 4.3 AJUSTE DE HIPERPARAMETROS
# ============================================================
# Criterio de seleccion: se optimiza f1-score de la clase positiva
# (alta_cianobacteria=1) en validacion cruzada. Se elige F1 en vez de
# accuracy porque, dado el desbalance ~81:1 (Ejercicio 2.4), accuracy
# es enganosa (un modelo trivial que siempre predice 0 ya obtendria
# ~98.8% de accuracy); F1 balancea precision y recall de la clase de
# interes sin descuidar ninguna de las dos por completo (la eleccion
# final entre precision/recall se discute con criterio ambiental en el
# Ejercicio 5.3).
#
# Por el tamano del conjunto de entrenamiento (~420k filas), la
# busqueda de hiperparametros se ejecuta sobre una submuestra
# estratificada (manteniendo la proporcion de clases) para que el
# tiempo de computo sea manejable; los mejores hiperparametros
# encontrados se reentrenan despues sobre el conjunto de entrenamiento
# completo para obtener el modelo final.

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

TAMANO_SUBMUESTRA_BUSQUEDA = 80_000
N_ITER_BUSQUEDA = 8
CV_BUSQUEDA = 3


def submuestra_estratificada(X, y, n, random_state=RANDOM_STATE):
    if n >= len(X):
        return X, y
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=n, random_state=random_state, stratify=y
    )
    return X_sub, y_sub


REJILLAS_HIPERPARAMETROS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    },
}


def construir_estimador_base(nombre, y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    if nombre == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    if nombre == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    if nombre == "xgboost":
        return XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(nombre)


def ajustar_hiperparametros(X_train, y_train):
    """
    Ejecuta RandomizedSearchCV (scoring="f1") sobre una submuestra
    estratificada del entrenamiento para cada modelo, y reentrena la
    mejor configuracion encontrada sobre el conjunto de entrenamiento
    completo.
    """

    X_sub, y_sub = submuestra_estratificada(X_train, y_train, TAMANO_SUBMUESTRA_BUSQUEDA)
    cv = StratifiedKFold(n_splits=CV_BUSQUEDA, shuffle=True, random_state=RANDOM_STATE)

    modelos_finales = {}
    filas_resumen = []

    for nombre, rejilla in REJILLAS_HIPERPARAMETROS.items():
        print(f"\nBuscando hiperparametros para {nombre}...")
        estimador = construir_estimador_base(nombre, y_sub)

        busqueda = RandomizedSearchCV(
            estimador,
            param_distributions=rejilla,
            n_iter=min(N_ITER_BUSQUEDA, np.prod([len(v) for v in rejilla.values()])),
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        busqueda.fit(X_sub, y_sub)

        print(f"Mejores hiperparametros ({nombre}): {busqueda.best_params_}")
        print(f"Mejor F1 en validacion cruzada (submuestra): {busqueda.best_score_:.4f}")

        modelo_final = construir_estimador_base(nombre, y_train)
        modelo_final.set_params(**busqueda.best_params_)
        modelo_final.fit(X_train, y_train)

        modelos_finales[nombre] = modelo_final
        filas_resumen.append({
            "modelo": nombre,
            "mejores_hiperparametros": str(busqueda.best_params_),
            "f1_cv_submuestra": busqueda.best_score_,
        })

    return modelos_finales, pd.DataFrame(filas_resumen)


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

    modelos_finales, resumen_hiperparametros = ajustar_hiperparametros(X_train, y_train)

    for nombre, modelo in modelos_finales.items():
        joblib.dump(modelo, os.path.join(CARPETA_MODELOS, f"{nombre}_final.joblib"))

    resumen_hiperparametros.to_csv(os.path.join(CARPETA_TABLAS, "hiperparametros_seleccionados.csv"), index=False)
    print(f"\nModelos finales (ajustados) guardados en -> {CARPETA_MODELOS}")
    print(resumen_hiperparametros.to_string(index=False))
