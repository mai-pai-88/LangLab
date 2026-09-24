# LangLab

LangLab is a document-aware chatbot and question-answering app for searching a collection of local documents. It can ingest PDFs, DOCX files, and plain text files, split them into chunks, build searchable indexes, and answer questions using an LLM plus retrieval. The project includes both a Streamlit chat interface and a command-line QA runner.

## What the app does

See the architectural overview here: [architecture.md](architecture.md).

LangLab works as a retrieval-augmented generation (RAG) system:

- It scans a document directory such as `doc_files`.
- It loads supported files: PDF, DOCX, and common text formats like `.txt`, `.md`, `.csv`, `.json`, `.yaml`, and similar.
- It splits each document into smaller chunks for better retrieval.
- It builds one of three retrieval modes:
  - `faiss`: semantic search with embeddings
  - `opensearch`: keyword retrieval via BM25
  - `hybrid`: combines both strategies
- It rephrases the user question to improve search quality.
- It classifies whether the question is about the document collection itself or about the content inside the documents.
- It uses a language model to answer the final question based on the retrieved passages.

The app also caches search indexes so repeated runs can reuse the built vector and keyword stores when the document content has not changed.



## Project structure

- `app.py`: core ingestion, indexing, retrieval, and CLI workflow
- `chat_app.py`: Streamlit-based chatbot interface
- `load_secrets.py`: AWS Secrets Manager helper for loading environment secrets
- `doc_files/`: default document store for the application
- `tests/`: project test suite

## Requirements

- Python 3.12+
- OpenAI API access via `OPENAI_API_KEY`
- Optional local OpenSearch instance at `http://localhost:9200` for keyword and hybrid retrieval modes
- AWS credentials if secret values are loaded from AWS Secrets Manager

Install dependencies with either:

```bash
pip install -r requirements.txt
```

or, if you are using the project environment management in this repo:

```bash
conda env create -f environment.yml
conda activate langlab
```

## Configuration

The application reads configuration from environment variables and `.env` files.

Common options:

```bash
export OPENAI_API_KEY="your-key"
export OPENSEARCH_URL="http://localhost:9200"
export RETRIEVER_MODE="hybrid"
```

If `OPENAI_API_KEY` is not already set, the application will attempt to load secrets from AWS Secrets Manager. See `load_secrets.py` for the default secret name and behavior.

## How to use it

### 1) Add documents

Put your PDFs, DOCX files, or text documents into the `doc_files` folder, or point the app to a different directory.

The default bundled sample is `doc_files/AAA-Identity-Management-Security.pdf`.

### 2) Start the Streamlit app

```bash
streamlit run chat_app.py
```

After the app starts, open the local Streamlit URL in a browser and submit questions about the loaded documents. The application will:

- answer content-based questions,
- classify general questions about the document collection itself,
- produce responses grounded in retrieved document passages and cached indexes.

### 3) Use the command-line app

The underlying document Q&A workflow can also be executed directly from Python:

```bash
python app.py "What is the title and chapters of this document?"
```

Use a different retrieval mode:

```bash
python app.py --retriever faiss "What is the main topic?"
python app.py --retriever opensearch "What is the main topic?"
python app.py --retriever hybrid "What is the main topic?"
```

Warm the document store indexes without asking a question:

```bash
python app.py --warm-index --retriever hybrid
```

This builds or reuses the index cache and exits.

## Retrieval modes

The application supports three retrieval strategies:

- `faiss`: semantic retrieval using embeddings
- `opensearch`: keyword-based BM25 search
- `hybrid`: combined FAISS and OpenSearch results using weighted ranking

`hybrid` is the default mode because it typically performs well across both concept-based and exact-term queries.

## Notes

- The application is intended for use with document collections rather than general web search.
- It reads only locally stored supported files from the configured directory.
- If the document set changes, the index manifest changes as well, which triggers cache refreshes.
- If OpenSearch is unavailable, keyword and hybrid retrieval modes may fail unless the service is running.

## Troubleshooting

Common verification steps:

- Ensure `OPENAI_API_KEY` is configured.
- Confirm that the expected files exist in `doc_files` or the selected document directory.
- Verify that OpenSearch is running when using the `opensearch` or `hybrid` retrieval mode.
- If AWS secrets are used, confirm that the relevant AWS credentials or environment variables are available.

For a quick validation, run:

```bash
python -c "from app import find_documents; print(find_documents('doc_files'))"
```

This confirms that the app can discover supported files in the document store.
