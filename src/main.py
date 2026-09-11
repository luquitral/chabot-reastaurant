"""Punto de entrada del asistente gastronomico."""
import os
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag_service import inicializar_vector_store
from clima_service import obtener_clima_actual
# Cargar variables de entorno (.env)
load_dotenv()

# 1. Contrato de datos de salida
class RecomendacionGastronomica(BaseModel):
    nombre_plato: str = Field(description="Nombre exacto del plato recomendado en la carta")
    precio_estimado: int = Field(description="Precio numérico del plato en CLP")
    maridaje_propuesto: Optional[str] = Field(default=None, description="Bebida o vino sugerido para acompañar")
    advertencia_alergenos: List[str] = Field(description="Alérgenos presentes o lista vacía")
    justificacion_sommelier: str = Field(description="Justificación detallada según las restricciones y gustos del comensal")

def construir_cadena():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta la variable GOOGLE_API_KEY. Configúrala en un archivo .env antes de iniciar el asistente."
        )

    parser = PydanticOutputParser(pydantic_object=RecomendacionGastronomica)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-3.6-flash"),
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Eres el Maitre y Sommelier digital de un restaurante de alta cocina.\n"
         "Recomienda la mejor opción basándote EXCLUSIVAMENTE en el contexto provisto de la carta.\n"
         "REGLA CRÍTICA DE SEGURIDAD: Revisa rigurosamente las alergias y restricciones dietéticas (celiaquía, vegetarianismo).\n"
         "Si un plato contiene un alérgeno incompatible con la petición, queda estrictamente descartado.\n\n"
         "Considera también el clima actual como criterio secundario: en días calurosos"
         "prioriza platos/bebidas frescos o ligeros; en días fríos o lluviosos prioriza opciones \n"
         "reconfortantes o calientes, siempre que sea coherente con la petición del comensal.\n\n"
         "Instrucciones de formato:\n{format_instructions}"),
        ("human",
         "Condición climática actual:\n{clima}\n\n"
         "Contexto de la carta disponible:\n{context}\n\n"
         "Consulta del comensal: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser

def main():
    base_dir = Path(__file__).resolve().parent.parent
    ruta_menu = base_dir / "data" / "menu.json"

    try:
        print("Inicializando base vectorial y modelos...")
        cadena = construir_cadena()
        retriever = inicializar_vector_store(ruta_menu)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"No se pudo iniciar el asistente: {error}")
        return

    print("Sistema listo. Escribe tu consulta (o 'salir' para finalizar):\n")

    clima_actual = obtener_clima_actual()
    print(f"{clima_actual}\n")

    while True:
        try:
            consulta = input("Comensal > ").strip()
            if not consulta:
                continue
            if consulta.lower() in ["salir", "exit", "quit"]:
                print("Cerrando asistente...")
                break

            t_inicio = time.perf_counter()

            # Búsqueda semántica de platos
            docs = retriever.invoke(consulta)
            contexto = "\n---\n".join([d.page_content for d in docs])

            # Inferencia y validación de esquema
            resultado: RecomendacionGastronomica = cadena.invoke({
                "context": contexto,
                "clima": clima_actual,
                "query": consulta
            })

            latencia = round(time.perf_counter() - t_inicio, 3)

            print(f"\n[Respuesta del Maitre] (Latencia: {latencia}s)")
            print(f"Plato recomendado : {resultado.nombre_plato} (${resultado.precio_estimado:,} CLP)")
            print(f"Maridaje sugerido : {resultado.maridaje_propuesto}")
            print(f"Alérgenos         : {', '.join(resultado.advertencia_alergenos) if resultado.advertencia_alergenos else 'Sin alérgenos reportados'}")
            print(f"Nota del Sommelier: {resultado.justificacion_sommelier}\n")
            print("-" * 60)

        except Exception as e:
            print(f"Error procesando la solicitud: {e}\n")

if __name__ == "__main__":
    main()