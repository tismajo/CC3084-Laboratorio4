import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CARPETA_TABLAS = "resultados/tablas"
CARPETA_FIGURAS = "resultados/figuras/comparacion"

os.makedirs(CARPETA_TABLAS, exist_ok=True)
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

RUTA_TEMPORAL = os.path.join(CARPETA_TABLAS, "indice_cianobacteria_temporal.csv")

if not os.path.exists(RUTA_TEMPORAL):
    raise FileNotFoundError(
        "No se encontro resultados/tablas/indice_cianobacteria_temporal.csv. "
        "Ejecuta primero temporaryAnalysis.py."
    )

df = pd.read_csv(RUTA_TEMPORAL)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.dropna(subset=["cyano_promedio"]).copy()

if df.empty:
    raise RuntimeError("La tabla temporal no contiene valores validos de cianobacteria.")

# Umbral comun para poder comparar ambos lagos bajo el mismo criterio.
umbral_comun = float(np.percentile(df["cyano_promedio"], 75))
df["fecha_critica_comun"] = df["cyano_promedio"] >= umbral_comun

registros = []

for lago, df_lago in df.groupby("lago"):
    df_lago = df_lago.sort_values("fecha").copy()
    fila_pico = df_lago.loc[df_lago["cyano_promedio"].idxmax()]

    cantidad_criticas = int(df_lago["fecha_critica_comun"].sum())
    frecuencia = 100 * cantidad_criticas / len(df_lago) if len(df_lago) > 0 else np.nan

    promedio = float(df_lago["cyano_promedio"].mean())
    desviacion = float(df_lago["cyano_promedio"].std(ddof=1)) if len(df_lago) > 1 else np.nan
    cv = 100 * desviacion / abs(promedio) if np.isfinite(desviacion) and promedio != 0 else np.nan

    registros.append({
        "lago": lago,
        "fechas_analizadas": int(len(df_lago)),
        "promedio_periodo": promedio,
        "mediana_periodo": float(df_lago["cyano_promedio"].median()),
        "p90_promedios": float(np.percentile(df_lago["cyano_promedio"], 90)),
        "maximo_promedio": float(fila_pico["cyano_promedio"]),
        "fecha_pico": fila_pico["fecha"].date().isoformat(),
        "desviacion_estandar": desviacion,
        "coeficiente_variacion_pct": cv,
        "umbral_comun_p75": umbral_comun,
        "fechas_criticas": cantidad_criticas,
        "frecuencia_fechas_criticas_pct": frecuencia,
    })

df_resumen = pd.DataFrame(registros)
ruta_resumen = os.path.join(CARPETA_TABLAS, "comparacion_lagos.csv")
df_resumen.to_csv(ruta_resumen, index=False)

# ============================================================
# 7.1 Evolucion temporal de ambos lagos
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))

for lago, df_lago in df.groupby("lago"):
    df_lago = df_lago.sort_values("fecha")
    ax.plot(df_lago["fecha"], df_lago["cyano_promedio"], marker="o", label=lago)

ax.axhline(
    umbral_comun,
    linestyle="--",
    linewidth=1.2,
    label=f"Umbral comun P75 = {umbral_comun:.2f}"
)
ax.set_title("Comparacion temporal de cianobacteria entre lagos")
ax.set_xlabel("Fecha")
ax.set_ylabel("Clorofila-a estimada")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(
    os.path.join(CARPETA_FIGURAS, "comparacion_temporal_lagos.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close(fig)

# ============================================================
# 7.2 Comparacion de intensidad
# ============================================================
lagos = list(df["lago"].drop_duplicates())
datos_boxplot = [
    df.loc[df["lago"] == lago, "cyano_promedio"].dropna().values
    for lago in lagos
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(datos_boxplot, tick_labels=lagos, showmeans=True)
ax.set_title("Distribucion de la intensidad promedio por lago")
ax.set_xlabel("Lago")
ax.set_ylabel("Clorofila-a estimada")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(
    os.path.join(CARPETA_FIGURAS, "boxplot_intensidad_lagos.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close(fig)

# Frecuencia de fechas criticas
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_resumen["lago"], df_resumen["frecuencia_fechas_criticas_pct"])
ax.set_title("Frecuencia relativa de fechas criticas por lago")
ax.set_xlabel("Lago")
ax.set_ylabel("Fechas criticas (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(
    os.path.join(CARPETA_FIGURAS, "frecuencia_fechas_criticas.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close(fig)

# Intensidad promedio
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_resumen["lago"], df_resumen["promedio_periodo"])
ax.set_title("Intensidad promedio de cianobacteria por lago")
ax.set_xlabel("Lago")
ax.set_ylabel("Clorofila-a estimada")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(
    os.path.join(CARPETA_FIGURAS, "intensidad_promedio_lagos.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close(fig)

print("\nComparacion entre lagos")
print("=" * 70)
print(f"Umbral comun relativo P75: {umbral_comun:.2f}")
print(df_resumen.to_string(index=False))
print("\nTabla guardada ->", ruta_resumen)
print("Figuras guardadas ->", CARPETA_FIGURAS)
print(
    "\nImportante: el punto 7.3 sobre posibles causas ambientales "
    "debe interpretarse en el informe usando estas evidencias y "
    "el contexto ambiental disponible."
)