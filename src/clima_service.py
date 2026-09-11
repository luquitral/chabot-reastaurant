import requests

"""Fuente externa del pipeline RAG: clima actual via Open-Meteo"""

# Codigos WMO de Open-Meteo 
_CODIGOS_CLIMA = {
    0: "despejado", 1: "mayormente despejado", 2: "parcialmente nublado",
    3: "nublado", 45: "con niebla", 48: "con niebla escarchada",
    51: "con llovizna ligera", 53: "con llovizna moderada", 55: "con llovizna intensa",
    61: "con lluvia ligera", 63: "con lluvia moderada", 65: "con lluvia intensa",
    71: "con nieve ligera", 73: "con nieve moderada", 75: "con nieve intensa",
    80: "con chubascos ligeros", 81: "con chubascos moderados", 82: "con chubascos violentos",
    95: "con tormenta electrica",
}

# Consulta el clima actual en Open-Meteo y lo retorna como texto de contexto
# Si la API falla, retorna un mensaje neutro

def obtener_clima_actual(latitud: float = -33.4489, longitud: float = -70.6693) -> str:

    try:
        respuesta = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitud,
                "longitude": longitud,
                "current": "temperature_2m,weather_code",
                "timezone": "America/Santiago",
            },
            timeout=5,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()["current"]

        temperatura = datos["temperature_2m"]
        codigo = datos["weather_code"]
        descripcion = _CODIGOS_CLIMA.get(codigo, "con condiciones variables")

        return (
            f"Clima actual en el restaurante: {temperatura}°C, {descripcion}."
        )
    except (requests.RequestException, KeyError, ValueError):
        return "Clima actual: no disponible (no considerar el clima en la recomendación)."