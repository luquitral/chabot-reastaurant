"""Este módulo se encarga de leer el JSON, crear los embeddings e inicializar el retriever de FAISS:"""

import json
from pathlib import Path
from typing import Any, List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def cargar_documentos_desde_json(ruta_json: Path) -> List[Document]:
    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_json}")

    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            platos: Any = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo de menú no contiene JSON válido: {error}") from error

    if not isinstance(platos, list):
        raise ValueError("El archivo de menú debe contener una lista de platos.")

    documentos = []
    campos_requeridos = {
        "id",
        "nombre",
        "categoria",
        "descripcion",
        "precio",
        "alergenos",
        "es_vegetariano",
        "es_celiaco",
        "maridaje_sugerido",
    }
    for indice, plato in enumerate(platos, start=1):
        if not isinstance(plato, dict) or not campos_requeridos.issubset(plato):
            raise ValueError(
                f"El plato #{indice} no tiene todos los campos requeridos del menú."
            )

        contenido = (
            f"Plato: {plato['nombre']}\n"
            f"Categoría: {plato['categoria']}\n"
            f"Descripción: {plato['descripcion']}\n"
            f"Precio: ${plato['precio']} CLP\n"
            f"Alérgenos: {', '.join(plato['alergenos']) if plato['alergenos'] else 'Ninguno'}\n"
            f"Apto para Vegetarianos: {'Sí' if plato['es_vegetariano'] else 'No'}\n"
            f"Apto para Celíacos: {'Sí' if plato['es_celiaco'] else 'No'}\n"
            f"Maridaje Recomendado: {plato['maridaje_sugerido']}"
        )
        doc = Document(
            page_content=contenido,
            metadata={"id": plato["id"], "nombre": plato["nombre"]}
        )
        documentos.append(doc)
    return documentos

def inicializar_vector_store(ruta_json: Path):
    documentos = cargar_documentos_desde_json(ruta_json)
    # Modelo de embeddings local ligero y de baja latencia
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_documents(documentos, embedding_model)
    return vector_db.as_retriever(search_kwargs={"k": 2})