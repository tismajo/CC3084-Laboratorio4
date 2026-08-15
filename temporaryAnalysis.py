import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

# fechas_atitlan = [
#     "2025-01-18", "2025-04-13", "2025-05-13", 
# ]
# """ "2025-07-17",
#     "2025-11-21", "2025-12-29", "2026-02-12", "2026-03-24",
#     "2026-04-13", "2026-04-28", "2026-07-22", """

# fechas_amatitlan = [
#     "2025-01-28", "2025-04-15", "2025-04-28", 
# ]

# """ "2025-11-24",
#     "2026-01-08", "2026-02-02", "2026-02-07", "2026-03-29",
#     "2026-04-13", "2026-04-28", "2026-06-19", """

fechas_atitlan = [
    "2025-01-18",
    "2025-04-13",
    "2025-05-13",
    "2025-07-17",
    "2025-11-21",
    "2025-12-29",
    "2026-02-12",
    "2026-03-24",
    "2026-04-13",
    "2026-04-28",
    "2026-07-22",
]

fechas_amatitlan = [
    "2025-01-28",
    "2025-04-15",
    "2025-04-28",
    "2025-11-24",
    "2026-01-08",
    "2026-02-02",
    "2026-02-07",
    "2026-03-29",
    "2026-04-13",
    "2026-04-28",
    "2026-06-19",
]

# ---------------------------------------------------------------------------
# 1. Indice promedio de cianobacteria por lago y por fecha
# ---------------------------------------------------------------------------
def calcular_promedio_por_fecha(carpeta_lago, fechas):
    registros = []
    for fecha in fechas:
        ruta = os.path.join(carpeta_lago, f"{fecha}_cyano.tif")
        if not os.path.exists(ruta):
            print(f"Falta {ruta}, se omite.")
            continue

        with rasterio.open(ruta) as src:
            datos = src.read(1)

        # NaN = pixeles fuera de agua, se ignoran en el promedio
        valores_agua = datos[~np.isnan(datos)]

        if valores_agua.size == 0:
            promedio, mediana, p90, pct_pixeles_agua = np.nan, np.nan, np.nan, 0.0
        else:
            promedio = float(np.mean(valores_agua))
            mediana = float(np.median(valores_agua))
            p90 = float(np.percentile(valores_agua, 90))
            pct_pixeles_agua = 100 * valores_agua.size / datos.size

        registros.append({
            "fecha": fecha,
            "cyano_promedio": promedio,
            "cyano_mediana": mediana,
            "cyano_p90": p90,
            "pct_area_agua": pct_pixeles_agua,
        })

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


df_atitlan = calcular_promedio_por_fecha("data/atitlan", fechas_atitlan)
df_amatitlan = calcular_promedio_por_fecha("data/amatitlan", fechas_amatitlan)

df_atitlan["lago"] = "Atitlan"
df_amatitlan["lago"] = "Amatitlan"

df_temporal = pd.concat([df_atitlan, df_amatitlan], ignore_index=True)

# Carpetas para guardar resultados
CARPETA_TABLAS = "resultados/tablas"
CARPETA_FIGURAS_TEMPORAL = "resultados/figuras/temporal"

os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS_TEMPORAL, exist_ok=True)

df_temporal.to_csv(
    os.path.join(CARPETA_TABLAS, "indice_cianobacteria_temporal.csv"),
    index=False
)

print("Tabla guardada en resultados/tablas/indice_cianobacteria_temporal.csv")
# print("Tabla guardada en resultados/indice_cianobacteria_temporal.csv")
print(df_temporal)

# ---------------------------------------------------------------------------
# 2. Grafico de linea por lago
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

for lago, df_lago in df_temporal.groupby("lago"):
    ax.plot(df_lago["fecha"], df_lago["cyano_promedio"], marker="o", label=lago)

ax.set_title("Evolucion temporal del indice de cianobacteria (chl-a)")
ax.set_xlabel("Fecha")
ax.set_ylabel("Clorofila-a estimada (mg/m³)")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    os.path.join(
        CARPETA_FIGURAS_TEMPORAL,
        "evolucion_temporal_cianobacteria.png"
    ),
    dpi=150
)
print("Grafico guardado en resultados/evolucion_temporal_cianobacteria.png")

# ---------------------------------------------------------------------------
# 3. Identificacion de picos y fechas criticas
# ---------------------------------------------------------------------------
def identificar_picos(df_lago, nombre_lago, umbral_percentil=75):
    df_valido = df_lago.dropna(subset=["cyano_promedio"])
    if df_valido.empty:
        print(f"[{nombre_lago}] Sin datos validos para identificar picos.")
        return

    umbral = np.percentile(df_valido["cyano_promedio"], umbral_percentil)
    picos = df_valido[df_valido["cyano_promedio"] >= umbral]

    print(f"\n[{nombre_lago}] Umbral de floracion (percentil {umbral_percentil}): {umbral:.2f}")
    print(f"[{nombre_lago}] Fechas criticas (>= umbral):")
    for _, fila in picos.iterrows():
        print(f"   {fila['fecha'].date()}  ->  chl-a promedio = {fila['cyano_promedio']:.2f}")

    fila_max = df_valido.loc[df_valido["cyano_promedio"].idxmax()]
    print(f"[{nombre_lago}] Pico maximo: {fila_max['fecha'].date()} "
          f"con chl-a = {fila_max['cyano_promedio']:.2f}")

for lago, df_lago in df_temporal.groupby("lago"):
    identificar_picos(df_lago, lago)

