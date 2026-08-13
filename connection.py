import openeo
import time
import os

# Conexión al backend de Copernicus Data Space Ecosystem
connection = openeo.connect("openeo.dataspace.copernicus.eu")

# Autenticación OAuth2 (abre navegador la primera vez)
connection.authenticate_oidc()

lago_atitlan = {
    "west": -91.326256,
    "east": -91.07151,
    "south": 14.5948,
    "north": 14.750979
}

lago_amatitlan = {
    "west": -90.638065,
    "east": -90.512924,
    "south": 14.412347,
    "north": 14.493799
}

fechas_atitlan = [
    "2025-01-18", "2025-04-13", "2025-05-13", "2025-07-17",
    "2025-11-21", "2025-12-29", "2026-02-12", "2026-03-24",
    "2026-04-13", "2026-04-28", "2026-07-22",
]

fechas_amatitlan = [
    "2025-01-28", "2025-04-15", "2025-04-28", "2025-11-24",
    "2026-01-08", "2026-02-02", "2026-02-07", "2026-03-29",
    "2026-04-13", "2026-04-28", "2026-06-19",
]

BANDAS = ["B03", "B04", "B08"]

def descargar_lago(nombre_lago, bbox, fechas, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    for fecha in fechas:
        ruta_salida = os.path.join(carpeta_salida, f"{fecha}.tif")

        if os.path.exists(ruta_salida):
            print(f"[{nombre_lago}] {fecha} ya existe, se omite.")
            continue

        try:
            cube = connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox,
                temporal_extent=[fecha, fecha],
                bands=BANDAS,
            )
            cube.download(ruta_salida)
            print(f"[{nombre_lago}] {fecha} descargado -> {ruta_salida}")

        except Exception as e:
            print(f"[{nombre_lago}] ERROR en {fecha}: {e}")

        # pausa breve para no saturar la API
        time.sleep(2)

descargar_lago("Atitlan", lago_atitlan, fechas_atitlan, "data/atitlan")
descargar_lago("Amatitlan", lago_amatitlan, fechas_amatitlan, "data/amatitlan")
print("Descarga finalizada ☺")
