# Chatbot de Restaurante — Asistente Gastronómico con RAG (EP1 ISY0101)

Asistente conversacional de consola que recomienda platos de un restaurante combinando:
- una **fuente interna** (la carta del restaurante, vectorizada con FAISS), y
- una **fuente externa** (el clima actual, vía Open-Meteo),

para generar una recomendación estructurada y validada (Pydantic), usando el modelo **Gemini** de Google como LLM.

## Arquitectura

Comensal (CLI) → Retriever FAISS (menú interno) + API de clima (externa) → Prompt + Gemini → Parser Pydantic → Respuesta estructurada.

Ver diagrama completo en `docs/arquitectura.png` (o en el informe técnico).

## Requisitos previos

- Python 3.12+
- Una API Key de Google AI Studio (https://aistudio.google.com/apikey)
- Conexión a internet.


## Instalación

```bash
git clone <url-del-repositorio>
cd chabot-reastaurant

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto con:

```
GOOGLE_API_KEY=tu_clave_de_google_ai_studio
GOOGLE_MODEL=gemini-3.6-flash
```

Cómo obtener `GOOGLE_API_KEY`:
1. Ve a https://aistudio.google.com/apikey
2. Inicia sesión con una cuenta de Google.
3. Crea una API key y cópiala en el `.env`.

`GOOGLE_MODEL` (opcional, por defecto gemini-3.6-flash): Permite especificar la versión del modelo de Gemini a utilizar, evitando inconsistencias o roturas si la API actualiza sus modelos predeterminados.

## Ejecución

```bash
python src/main.py
```

Al iniciar, el sistema:
1. Descarga (la primera vez) y carga el modelo de embeddings local.
2. Indexa la carta (`data/menu.json`) en una base vectorial FAISS en memoria.
3. Consulta el clima actual (Open-Meteo) una vez, como contexto externo para toda la sesión.
4. Queda esperando consultas en el prompt `Comensal >`.

Ejemplo de uso:
```
Comensal > soy celíaco, ¿qué me recomiendas de plato de fondo?
```

Escribe `salir` para terminar la sesión.

## Estructura del proyecto

```
chatbot-restaurant/
├── data/
│   └── menu.json           # Fuente interna: carta del restaurante
├── src/
│   ├── main.py              # Punto de entrada, orquesta el flujo RAG
│   ├── rag_service.py       # Carga y vectorización del menú (FAISS)
│   └── clima_service.py     # Fuente externa: clima actual (Open-Meteo)
├── requirements.txt
├── .env                     # Credenciales (no versionado)
└── README.md
```

## Notas técnicas

- El clima se consulta **una vez por sesión** (no en cada mensaje), para no depender de la disponibilidad de la API externa en cada consulta.
- Si la API de clima falla, el sistema continúa funcionando y simplemente indica al modelo que no considere el clima.
- La salida del LLM está forzada a un esquema fijo mediante `PydanticOutputParser`, por lo que toda recomendación es validable programáticamente.

## Uso de inteligencia artificial

Se utilizó un asistente de IA (Claude, Anthropic) como apoyo para depurar errores de entorno/dependencias, diseñar la integración de la fuente de datos externa y estructurar la documentación de este proyecto. Ver detalle en el informe técnico, sección "Uso de inteligencia artificial".
