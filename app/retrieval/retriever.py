from dataclasses import dataclass
from typing import List, Optional

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document

from app.ingest.embed_qdrant import EmbeddingSelfQuery
from app.retrieval.self_query import (
    document_content_description,
    metadata_field_info,
)


@dataclass
class SelfQueryConfig:
    """
    Configuração para o SelfQueryRetriever.

    Esta classe armazena as configurações usadas ao construir o retriever, como o nome da coleção no banco de dados Qdrant e o número de resultados desejados.

    Atributos:
        collection_name (str): Nome da coleção do Qdrant (padrão: "sumulas_jornada").
        k (int): Número de resultados a serem retornados na consulta (padrão: 10).
    """

    collection_name: str = "sumulas_jornada"
    k: int = 10


def build_self_query_retriever(cfg: SelfQueryConfig) -> SelfQueryRetriever:
    """
    Cria o SelfQueryRetriever sobre o QdrantVectorStore.

    Esta função configura e retorna um `SelfQueryRetriever` utilizando a configuração fornecida.
    O retriever é configurado para realizar buscas usando a coleção do Qdrant e o modelo de LLM fornecido pelo embedder.

    Args:
        cfg (SelfQueryConfig): Configuração contendo o nome da coleção e o número de resultados.

    Returns:
        SelfQueryRetriever: Um objeto configurado para realizar consultas no banco de dados Qdrant com base na configuração fornecida.
    """
    embedder = EmbeddingSelfQuery()
    vectorstore = embedder.get_qdrant_vector_store(cfg.collection_name)

    retriever = SelfQueryRetriever.from_llm(
        llm=embedder.llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        search_kwargs={"k": cfg.k},
    )
    return retriever


def search(
    query: str,
    cfg: Optional[SelfQueryConfig] = None,
) -> List[Document]:
    """
    Consulta usando self-query: o LLM infere termos SEMÂNTICOS e também FILTROS de metadado.

    Esta função realiza uma consulta no banco de dados Qdrant usando um modelo de linguagem (LLM) para inferir termos semânticos
    e filtros de metadados. O resultado é uma lista de documentos que correspondem à consulta.

    Args:
        query (str): A consulta textual a ser realizada.
        cfg (Optional[SelfQueryConfig]): A configuração personalizada para o retriever. Se não fornecido, usa a configuração padrão.

    Returns:
        List[Document]: Lista de documentos (`Document`) que correspondem à consulta, incluindo metadados e conteúdo relevante.
    """
    cfg = cfg or SelfQueryConfig()
    retriever = build_self_query_retriever(cfg)
    # .invoke() retorna List[Document]
    return retriever.invoke(query)
