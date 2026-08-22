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


# ============================================================
# 10.2 LIMITACIONES DEL MODELO
# ============================================================

def limitaciones(distribucion_lago, bloques):
    fila_atitlan = distribucion_lago[distribucion_lago["lago"] == "Atitlan"].iloc[0]
    n_bloques_atitlan = bloques[bloques["lago"] == "Atitlan"]["n_bloques"].iloc[0]
    n_bloques_amatitlan = bloques[bloques["lago"] == "Amatitlan"]["n_bloques"].iloc[0]

    texto = f"""
--- 10.2 Limitaciones encontradas durante el desarrollo del modelo ---

  - Cantidad y espaciamiento de fechas: solo 11 fechas por lago,
    distribuidas de forma irregular a lo largo de ~1.5 anios (algunas
    con varios meses de separacion), lo que limita la capacidad del
    modelo para capturar la dinamica temporal completa de las
    floraciones (aparicion, pico, disipacion) y probablemente deja
    fuera eventos de floracion breves entre fechas de captura.

  - Nubosidad y pixeles invalidos: varias fechas muestran menos
    pixeles de agua validos que otras (ver Ejercicio 1.4), reflejo de
    interferencia de nubes o condiciones atmosfericas, lo que reduce
    de forma desigual la cantidad de evidencia disponible por fecha.

  - Resolucion espacial reducida (50m vs 20m de la Parte I): fue una
    decision practica (Ejercicio 1.6) para mantener manejable el
    volumen de descarga via API; esto pudo suavizar/perder patrones
    finos de floracion que si eran visibles en los mapas de mayor
    resolucion de la Parte I.

  - Diferencias extremas entre lagos: Atitlan tiene apenas
    {int(fila_atitlan['n_positivos'])} pixeles positivos en toda la
    serie (tasa de positivos {fila_atitlan['tasa_positivos']:.3f}%)
    frente a una tasa mucho mayor en Amatitlan, lo que hace que
    cualquier metrica agregada este dominada por Amatitlan y que el
    modelo tenga muy poca senal para aprender floraciones raras en
    Atitlan.

  - Metodologia de validacion: la validacion aleatoria (Ejercicio 5)
    sobreestima el desempeno respecto a la validacion espacial
    (Ejercicio 6, {n_bloques_atitlan} bloques en Atitlan y
    {n_bloques_amatitlan} en Amatitlan) debido a autocorrelacion
    espacial entre pixeles vecinos; y el modelo no generaliza entre
    lagos (Ejercicio 7), por lo que las metricas reportadas solo son
    validas para el lago en el que se entreno el modelo.

  - cyano_index como proxy, no medicion directa: el indice de
    cianobacteria usado como base de la variable respuesta es un
    proxy espectral (NDCI, Mishra & Mishra 2012) y no una medicion de
    laboratorio de microcistinas o conteo celular real, por lo que
    hereda las limitaciones de cualquier estimacion remota
    (posibles valores extremos no acotados, ver Ejercicio 1.6 decision
    8).
"""
    print(texto)
    return texto


if __name__ == "__main__":
    metricas, cv_comparacion, generalizacion, distribucion_lago, bloques = cargar_resumenes()

    texto_101 = analisis_capacidad_modelo(metricas, cv_comparacion)

    with open(os.path.join(CARPETA_TABLAS, "conclusiones_10_1.txt"), "w", encoding="utf-8") as f:
        f.write(texto_101)

    texto_102 = limitaciones(distribucion_lago, bloques)

    with open(os.path.join(CARPETA_TABLAS, "conclusiones_10_2.txt"), "w", encoding="utf-8") as f:
        f.write(texto_102)
