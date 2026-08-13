import openeo
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
    "2026-04-13", "2026-04-28", "2026-07-22"
]

datacube = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=lago_atitlan,
    temporal_extent=[fechas_atitlan[0], fechas_atitlan[-1]],
    bands=["B03", "B04", "B08"]  # NDWI, NDVI, cianobacteria
)

os.makedirs("data/atitlan", exist_ok=True)

for fecha in fechas_atitlan:
    cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=lago_atitlan,
        temporal_extent=[fecha, fecha],
        bands=["B03", "B04", "B08"]
    )
    cube.download(f"data/atitlan/{fecha}.tif")
    print(f"Descargado {fecha}")

print(connection.describe_collection("SENTINEL2_L2A"))