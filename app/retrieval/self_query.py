from langchain.chains.query_constructor.schema import AttributeInfo

"""
Lista de informações de atributos (metadados) que descrevem cada campo nas súmulas do Tribunal de Contas de Minas Gerais.

Cada item na lista é um objeto `AttributeInfo`, que descreve um campo específico nos documentos, incluindo o nome do campo (`name`), a descrição detalhada sobre o uso do campo (`description`), e o tipo de dado esperado para o campo (`type`). Esses atributos serão usados para definir as regras de busca e comparação em consultas sobre as súmulas.

A lista contém as seguintes informações:

1. **num_sumula**: Número da súmula, representado como uma string.
2. **status_atual**: O status atual da súmula (ex: 'VIGENTE', 'REVOGADA', 'ALTERADA'), representado como uma string.
3. **data_status**: Data de status da súmula no formato 'DD/MM/AA', representada como uma string.
4. **data_status_ano**: Ano de publicação da súmula, representado como um número inteiro.
5. **pdf_name**: Nome do arquivo PDF de origem, representado como uma string.
6. **chunk_type**: Tipo do chunk do documento (ex: 'conteudo_principal', 'referencias_normativas', 'precedentes'), representado como uma string.
7. **chunk_index**: Índice do chunk dentro do documento, representado como um número inteiro.

Cada `AttributeInfo` é um objeto com a seguinte estrutura:
    - **name** (str): Nome do campo de metadado.
    - **description** (str): Descrição detalhada sobre o campo e como ele deve ser utilizado.
    - **type** (str): Tipo de dado esperado para o campo (ex.: "string", "integer").

Tipo de `metadata_field_info`: `List[AttributeInfo]`
"""

metadata_field_info = [
    AttributeInfo(
        name="num_sumula",
        description=(
            "- Número da súmula (ex.: '70'). Texto simples, sem prefixo.\n"
            "- Sempre filtre pelo número da súmula quando o usuário perguntar exclusivamente usando o número."
        ),
        type="string",
    ),
    AttributeInfo(
        name="status_atual",
        description="Status atual da súmula (ex.: 'VIGENTE', 'REVOGADA', 'ALTERADA', etc.).",
        type="string",
    ),
    AttributeInfo(
        name="data_status",
        description=(
            "Data textual no formato 'DD/MM/AA' (string). Ex.: '07/04/14'.\n"
        ),
        type="string",
    ),
    AttributeInfo(
        name="data_status_ano",
        description=(
            "Ano da publicação no formato 'AAAA' (integer). Ex.: '2014'.\n"
            "- Você PODE usar operadores de comparação (lt, gt, lte, gte) e igualdade (eq).\n"
            "- Para anos (ex.: 'antes de 2010'), interprete como comparações sobre datas, mesmo que o campo seja string.\n"
            "- Se o usuário disser 'antes de AAAA', use 'lt' e considere o começo do ano AAAA como limite.\n"
            "- Se o usuário disser 'depois de AAAA', use 'gt' e considere o fim do ano AAAA como limite.\n"
        ),
        type="integer",
    ),
    AttributeInfo(
        name="pdf_name",
        description="Nome do arquivo PDF de origem (ex.: 'Sumula_70.pdf').",
        type="string",
    ),
    AttributeInfo(
        name="chunk_type",
        description="Tipo do chunk: 'conteudo_principal', 'referencias_normativas' ou 'precedentes'.",
        type="string",
    ),
    AttributeInfo(
        name="chunk_index",
        description="Índice do chunk no documento.",
        type="integer",
    ),
]
document_content_description = """
    Coleção de trechos (chunks) de súmulas do Tribunal de Contas de Minas Gerais, 
    cada uma com metadados como número (num_sumula), status (status_atual), 
    data textual (data_status, formato 'DD/MM/AA'), nome do arquivo (pdf_name) e tipo de trecho (chunk_type).\n\n
"""
