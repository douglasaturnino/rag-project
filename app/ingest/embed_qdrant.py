from langchain.chat_models import init_chat_model
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.utils.settings import settings


class EmbeddingSelfQuery:
    """
    Classe que gerencia a inicialização do modelo de linguagem, a configuração do cliente Qdrant e o modelo de embeddings para realizar consultas.

    A classe fornece métodos para obter um `QdrantVectorStore`, que é usado para armazenar e recuperar embeddings de vetores
    para a realização de consultas baseadas em semântica.

    Attributes:
        llm (BaseChatModel): O modelo de linguagem inicializado para gerar respostas baseadas em consultas.
        client (QdrantClient): O cliente Qdrant usado para interagir com o banco de dados Qdrant.
        model (FastEmbedEmbeddings): O modelo de embeddings que converte texto em representações vetoriais.
    """

    def __init__(self) -> None:
        """
        Inicializa o modelo de linguagem, o cliente Qdrant e o modelo de embeddings.

        O modelo de linguagem é inicializado com configurações definidas nas variáveis de ambiente.
        O cliente Qdrant é configurado com os parâmetros de host e porta definidos nas configurações.
        O modelo de embeddings é inicializado para conversões de texto em vetores de alta qualidade.
        """

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
        """
        Retorna uma instância do `QdrantVectorStore` configurada com o cliente Qdrant,
        o modelo de embeddings e os parâmetros da coleção.

        Args:
            collection_name (str): Nome da coleção onde os vetores serão armazenados no Qdrant.

        Returns:
            (QdrantVectorStore): A instância configurada do `QdrantVectorStore`.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.model,
            sparse_vector_name="text-sparse",
            vector_name="text-dense",
        )
