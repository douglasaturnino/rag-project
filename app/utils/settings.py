import os
from platform import system

from dotenv import find_dotenv, load_dotenv


class Settings:
    """
    Classe responsável por carregar as configurações do sistema, variáveis de ambiente e definir parâmetros específicos
    para a aplicação, como as configurações do Qdrant, modelo e chaves da API.

    A classe carrega automaticamente o arquivo `.env` e disponibiliza os valores como atributos de classe.
    """

    load_dotenv()

    ENV_FILE = find_dotenv()
    SYSTEM = system()

    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = os.getenv("QDRANT_PORT")
    TEMPERATURE = os.getenv("TEMPERATURE")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME")


settings = Settings()
