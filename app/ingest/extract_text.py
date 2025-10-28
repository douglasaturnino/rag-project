import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from markitdown import MarkItDown
from qdrant_client.http.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from app.graph.prompt import PROMPT_EXTRACT
from app.ingest.embed_qdrant import EmbeddingSelfQuery

md = MarkItDown()


def process_pdf_file(
    file_path: str, embedder: EmbeddingSelfQuery
) -> List[Dict[str, Any]]:
    """
    Usa o LLM interno do embedder para extrair metadados e dividir em até 3 chunks
    """
    pdf_name = os.path.basename(file_path)
    result = md.convert(str(file_path))
    text_content = result.text_content or ""

    # Prompt de extração
    prompt = PROMPT_EXTRACT.format(
        pdf_name=pdf_name, text_content=text_content[:12000]
    )

    try:
        response = embedder.llm.invoke(prompt)
        json_text = (
            re.sub(r"```[\w-]*", "", response.content)
            .replace("```", "")
            .strip()
        )
        data = json.loads(json_text)

        metadados = data.get("metadados", {})
        chunks = data.get("chunks", {})

        processed = []
        for idx, (tipo, texto) in enumerate(chunks.items()):
            if not texto or idx >= 3:
                continue
            metadata = {
                "num_sumula": metadados.get("num_sumula"),
                "data_status": metadados.get("data_status"),
                "data_status_ano": int(metadados.get("data_status_ano")),
                "status_atual": metadados.get("status_atual"),
                "pdf_name": metadados.get("pdf_name", pdf_name),
                "chunk_type": tipo,
                "chunk_index": idx,
            }
            processed.append({"text": texto.strip(), "metadata": metadata})

        return processed

    except Exception as e:
        logger.error(f"⚠️ Erro ao processar {pdf_name}: {e}")
        return []


def create_collection_if_not_exists(
    embedder: EmbeddingSelfQuery, collection: str
):
    # Cria coleção se não existir

    if embedder.client.collection_exists(collection_name=collection):
        logger.info(f"Coleção '{collection}' já existe.")
        return

    embedder.client.create_collection(
        collection_name=collection,
        vectors_config={
            "text-dense": VectorParams(
                size=embedder.model.model.embedding_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams()  # sem size para esparso
        },
    )
    logger.info(f"Coleção '{collection}' criada.")


def main(
    collection: str = "sumulas_jornada",
    pasta_pdfs: str = "sumulas",
):
    embedder = EmbeddingSelfQuery()

    create_collection_if_not_exists(embedder, collection)

    vector_store = embedder.get_qdrant_vector_store(collection)
    pdf_files = list(Path(pasta_pdfs).glob("*.pdf"))
    if not pdf_files:
        logger.info("Nenhum PDF encontrado na pasta.")
        return

    total_chunks = 0
    for pdf_file in pdf_files:
        logger.debug(f"Processando {pdf_file.name}.")
        chunks = process_pdf_file(str(pdf_file), embedder)
        if not chunks:
            continue
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        total_chunks += len(chunks)
        logger.debug(f"{pdf_file.name} processada.")

    logger.success(
        f"✅ {len(pdf_files)} PDFs processados. {total_chunks} chunks inseridos no Qdrant."
    )


if __name__ == "__main__":
    main()
