"""
Parte 2 - Ejercicio 10: Analisis y conclusiones.

Sintetiza los resultados de todos los ejercicios anteriores (4-9) para
responder si el modelo desarrollado es util como herramienta de apoyo,
que limitaciones tiene y que datos adicionales podrian mejorarlo.
"""

import os
import pandas as pd

CARPETA_TABLAS = "parte2/resultados/tablas"


def cargar_resumenes():
    metricas = pd.read_csv(os.path.join(CARPETA_TABLAS, "comparacion_modelos_metricas.csv"))
    cv_comparacion = pd.read_csv(os.path.join(CARPETA_TABLAS, "comparacion_aleatoria_vs_espacial.csv"))
    generalizacion = pd.read_csv(os.path.join(CARPETA_TABLAS, "generalizacion_entre_lagos.csv"))
    distribucion_lago = pd.read_csv(os.path.join(CARPETA_TABLAS, "distribucion_respuesta_por_lago.csv"))
    bloques = pd.read_csv(os.path.join(CARPETA_TABLAS, "resumen_bloques_espaciales.csv"))
    return metricas, cv_comparacion, generalizacion, distribucion_lago, bloques


# ============================================================
# 10.1 CAPACIDAD DEL MODELO COMO HERRAMIENTA DE APOYO
# ============================================================

def analisis_capacidad_modelo(metricas, cv_comparacion):
    xgb = metricas[metricas["modelo"] == "xgboost"].iloc[0]
    xgb_cv_espacial_recall = cv_comparacion[
        (cv_comparacion["modelo"] == "xgboost") & (cv_comparacion["metrica"] == "recall")
    ]["espacial"].iloc[0]

    texto = f"""
--- 10.1 Capacidad del modelo como herramienta de apoyo ---

El modelo final (XGBoost, seleccionado en el Ejercicio 5.3 por
priorizar recall) alcanza en el conjunto de prueba (division aleatoria
70/30): recall={xgb['recall']:.3f}, precision={xgb['precision']:.3f},
F1={xgb['f1']:.3f}, ROC-AUC={xgb['roc_auc']:.3f}. Bajo validacion
espacial (GroupKFold por bloque de 1km, Ejercicio 6), el recall se
mantiene alto ({xgb_cv_espacial_recall:.3f}), lo que indica que el
modelo SI tiene capacidad util como herramienta de APOYO al monitoreo:
detecta la gran mayoria de las zonas con alta presencia de
cianobacteria (pocos falsos negativos, el error mas costoso desde el
punto de vista de salud publica segun el Ejercicio 5.3), a costa de
generar tambien falsas alarmas que un equipo humano deberia poder
descartar con una verificacion rapida.

Sin embargo, esta capacidad es claramente condicional: el modelo
funciona bien DENTRO de un mismo lago y con datos de la misma
distribucion con la que fue entrenado, pero el Ejercicio 7 mostro que
NO generaliza de un lago a otro (recall tan bajo como 0.001-0.19 en
los experimentos cruzados). Por lo tanto, el modelo es util como
apoyo al monitoreo SIEMPRE que se entrene y aplique de forma
especifica para cada lago, y no deberia usarse como sustituto de
muestreos fisicos de confirmacion, sino como sistema de priorizacion
de zonas a inspeccionar.
"""
    print(texto)
    return texto


if __name__ == "__main__":
    metricas, cv_comparacion, generalizacion, distribucion_lago, bloques = cargar_resumenes()

    texto_101 = analisis_capacidad_modelo(metricas, cv_comparacion)

    with open(os.path.join(CARPETA_TABLAS, "conclusiones_10_1.txt"), "w", encoding="utf-8") as f:
        f.write(texto_101)
