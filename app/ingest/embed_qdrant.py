from langchain.chat_models import init_chat_model
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.utils.settings import settings


class EmbeddingSelfQuery:
    def __init__(self) -> None:
        self.llm = init_chat_model(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
        )
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=120,
        )

        self.model = FastEmbedEmbeddings(model_name=settings.EMBEDDINGS_NAME)

    def get_qdrant_vector_store(
        self, collection_name: str
    ) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.model,
            sparse_vector_name="text-sparse",
            vector_name="text-dense",
        )
