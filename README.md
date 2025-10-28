# Pipeline RAG completa: Ingestão, Consulta e Obserabilidade

Este projeto demonstra como construir um **pipeline completo de Recuperação e Geração (RAG)** usando:
- **LangChain** → integração entre LLMs e bancos vetoriais.  
- **LangGraph** → orquestração de nós e controle de fluxo.  
- **Langfuse** → observabilidade e logging de cada execução.  
- **Qdrant** → vetor store local (armazenamento vetorial).  
- **Streamlit** → interface interativa para o usuário final.

O objetivo é mostrar, de forma prática, como montar uma aplicação jurídica capaz de responder perguntas sobre **súmulas do Tribunal de Contas de Minas Gerais (TCEMG)** com base em documentos PDF indexados.

---

## ⚙️ Estrutura do Projeto

```
rag-project/
├── app/
│   ├── graph/
│   │   ├── rag_graph.py      # Grafo principal do LangGraph
│   │   └── prompt.py         # Prompts e templates do LLM
│   ├── ingest/
│   │   ├── embed_qdrant.py   # Embeddings e conexão com Qdrant
│   │   └── extract_text.py   # Extração de texto e metadados dos PDFs
│   ├── retrieval/
│   │   ├── retriever.py      # SelfQueryRetriever + Qdrant
│   │   └── self_query.py     # Definição dos metadados e filtros
│   ├── app.py                # Interface Streamlit
│   └── settings.py           # Configurações globais do app
│
├── requirements.txt
├── README.md
└── Dockerfile (opcional)
```

---

## 🧩 Arquitetura Geral

![alt text](image-1.png)


---

## 🧠 Conceitos-Chave

- **RAG (Retrieval-Augmented Generation)**  
  Combina busca semântica com geração de texto. O modelo recupera contexto real antes de gerar a resposta.  

- **Self-Query Retriever**  
  Permite que o próprio modelo **entenda a pergunta** e monte filtros estruturados automaticamente (por exemplo, filtrar por `status_atual` ou `data_status`).  

- **LangGraph**  
  Controla o fluxo entre os nós (`retrieve → generate → END`) e o estado global.  

- **Langfuse**  
  Garante **observabilidade completa**: rastreamento de spans, tokens e métricas de cada execução.

---

## 🚀 Como Executar o Projeto Localmente

### 1️⃣ Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop)  
- Python 3.11+  
- Conta gratuita em:
  - [OpenAI Platform](https://platform.openai.com/settings/organization/api-keys)
  - [Langfuse Cloud](https://us.cloud.langfuse.com/)
- Clonar o repositório:

```bash
git clone https://github.com/seu-usuario/rag-project.git
cd rag-project
```

---

### 2️⃣ Subir o Qdrant localmente

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

---

### 3️⃣ Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxx
LANGFUSE_PUBLIC_KEY=pk-xxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-xxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://us.cloud.langfuse.com
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

### 4️⃣ Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate     # (ou .venv\Scripts\activate no Windows)
pip install -r requirements.txt
```

---

### 5️⃣ Ingerir os documentos

Coloque seus arquivos PDF na pasta `sumulas/`.  
Você pode baixar documentos oficiais do TCE-MG aqui:  
🔗 [https://www.tce.mg.gov.br/Noticia/Detalhe/67](https://www.tce.mg.gov.br/Noticia/Detalhe/67)

Depois execute:

```bash
python app/ingest/extract_text.py
```

Isso criará a coleção `sumulas_jornada` no Qdrant, extraindo texto e metadados automaticamente.

---

### 6️⃣ Executar o app

```bash
streamlit run app/app.py
```

Abra o navegador em **http://localhost:8501**.

---

## 🧮 Como o Projeto Lida com Datas

> 🗓️ O Qdrant só suporta comparações (`<`, `>`) em valores numéricos, não em strings.

Como os metadados vinham no formato `DD/MM/AA`, as comparações lexicográficas não funcionam:
```
'01/01/10' > '31/12/09'  # (errado, pois compara texto)
```

**Solução:**  
Durante a ingestão, o campo `data_status` é convertido para **ano (inteiro)**, ex.:

```
"07/04/14" → 2014
```

Isso permite aplicar filtros numéricos no SelfQueryRetriever:

```python
AttributeInfo(
    name="data_status",
    description="Ano da publicação (inteiro). Ex.: 2014. Use lt/lte/gt/gte para comparar anos.",
    type="integer",
)
```
## 💡 Dica sobre Datas e Comparações

> O Qdrant Translator geralmente só permite **filtros de igualdade (==)**.  
> Filtros de comparação (`<`, `>`) em strings não funcionam para datas,  
> a menos que o formato seja **ISO 8601 (YYYY-MM-DD)**.  
> Como este projeto usa **DD/MM/AA**, é essencial armazenar o campo `data_status` como inteiro (ano).

---

## 📊 Observabilidade com Langfuse

Cada execução é rastreada no painel da Langfuse, incluindo:

- Prompt e contexto usados  
- Tokens consumidos  
- Tempo de resposta  
- Cadeia de nós (`retrieve → generate`)  

🔗 [Acesse o painel Langfuse Cloud](https://us.cloud.langfuse.com/)  
para visualizar seus **traces em tempo real**.

---

## 🔗 Referências Técnicas

- 📄 [MarkItDown (Microsoft)](https://github.com/microsoft/markitdown) – extração de texto e metadados dos PDFs.  
- 💾 [Qdrant LangChain API](https://python.langchain.com/api_reference/qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html) – integração vetorial.  
- 🔍 [SelfQueryRetriever Docs](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.self_query.base.SelfQueryRetriever.html)  
- ⚙️ [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/)  

---
