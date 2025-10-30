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
    Processa um arquivo PDF, extraindo metadados e dividindo o conteúdo em até 3 chunks, retornando uma lista de dicionários.

    A função usa o modelo de linguagem do `embedder` para extrair os metadados da súmula e dividir o texto em até três partes principais:

    1. conteudo_principal: O conteúdo principal do documento até a seção de "REFERÊNCIAS NORMATIVAS".

    2. referencias_normativas: O conteúdo entre "REFERÊNCIAS NORMATIVAS:" e "PRECEDENTES:".

    3. precedentes: O conteúdo após a seção de "PRECEDENTES:".

    Args:
        file_path (str): Caminho para o arquivo PDF que será processado.
        embedder (EmbeddingSelfQuery): O objeto que contém o modelo de linguagem usado para extrair os metadados e chunks.

    Returns:
        processed (List[Dict[str, Any]]): Uma lista de dicionários contendo os textos dos chunks e seus respectivos metadados.
            Cada dicionário tem a estrutura:
            ```json
            {
                "text": <texto do chunk>,
                "metadata": {
                    "num_sumula": <número da súmula>,
                    "data_status": <data de status>,
                    "data_status_ano": <ano da data de status>,
                    "status_atual": <status atual>,
                    "pdf_name": <nome do arquivo PDF>,
                    "chunk_type": <tipo do chunk>,
                    "chunk_index": <índice do chunk>
                }
            }
            ```
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
) -> None:
    """
    Cria uma coleção no Qdrant se ela não existir, configurando os parâmetros para vetores densos e esparsos.

    A função verifica se a coleção especificada já existe no Qdrant. Se não existir, cria a coleção com as configurações adequadas
    para vetores densos e esparsos, usando o modelo de embeddings configurado no `embedder`.

    Args:
        embedder (EmbeddingSelfQuery): O objeto que contém o cliente Qdrant e o modelo de embeddings.
        collection (str): O nome da coleção a ser criada ou verificada.
    """

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
) -> None:
    """
    Função principal que orquestra o processamento de arquivos PDF, a criação de coleções no Qdrant e a adição de vetores.

    Esta função:
    1. Verifica e cria a coleção no Qdrant se não existir.
    2. Processa os arquivos PDF da pasta especificada, extraindo os chunks e metadados.
    3. Adiciona os textos e metadados ao Qdrant.

    Args:
        collection (str, opcional): Nome da coleção do Qdrant. Padrão é "sumulas_jornada".
        pasta_pdfs (str, opcional): Caminho da pasta contendo os arquivos PDF a serem processados. Padrão é "sumulas".

    """

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
