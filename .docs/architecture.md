# LangLab Architecture

LangLab is a local-first document question-answering application. It combines
Streamlit for the chat experience, LangChain for retrieval orchestration,
OpenAI-compatible models for embeddings and answers, FAISS for semantic
search, and an optional OpenSearch service for keyword or hybrid retrieval.

The design prioritizes a responsive interactive experience for a curated
collection of roughly 10 to 5,000 documents, usually producing thousands to
low hundreds of thousands of 500-character chunks when adequate memory and
OpenSearch capacity are available. The practical limit is total extractable
text and chunk count, rather than raw file size; a scanned PDF with little
text costs far less to index than a text-heavy manual of the same size.
Retrieval options remain explicit and testable.

## Architecture at a Glance

![LangLab component architecture](architecture.svg)

*Open [architecture.html](architecture.html) in a browser for the interactive
version with search, focus, and relationship tracing.*

## Features and Benefits

| Feature | Implementation | Benefit |
| --- | --- | --- |
| Multi-format document ingestion | PDF, DOCX, and supported UTF-8 text formats are discovered recursively in `doc_files`. | A single collection can include source material, notes, exports, and documentation without a conversion pipeline. |
| Deterministic document selection | The default CLI PDF path is resolved relative to `app.py`; document discovery is sorted. | Startup does not depend on the shell working directory, and ingestion order is repeatable. |
| Chunked retrieval | Documents are split into 500-character chunks with 50-character overlap. | Small chunks improve retrieval precision; overlap preserves context at chunk boundaries. |
| Semantic search | FAISS stores embeddings generated through `OpenAIEmbeddings`. | Finds relevant material even when user phrasing differs from the source text. |
| Keyword search | OpenSearch stores chunks in a BM25-compatible, manifest-addressed index generation behind a stable alias. | Handles exact terms, identifiers, acronyms, and product names without rebuilding an unchanged keyword index. |
| Hybrid retrieval | `EnsembleRetriever` merges FAISS and keyword results with equal reciprocal-rank-fusion weights. | Reduces the blind spots of either search method alone. |
| User-facing retrieval behavior | Chat questions are routed through the selected retrieval mode; users do not interact with FAISS or OpenSearch directly. | FAISS helps the chatbot recognize meaning across different wording, while OpenSearch helps it locate exact terms and identifiers. |
| Retrieval modes | `RETRIEVER_MODE` and CLI options select `faiss`, `opensearch`, or `hybrid`. | Lets an operator trade embedding cost and semantic recall against fast exact-term matching. |
| Versioned retrieval-index cache | A SHA-256 manifest records each document's relative path and content hash. A matching manifest reuses the persisted FAISS index in `.langlab` and the matching OpenSearch generation; any add, removal, or content change builds new indexes. | Unchanged collections avoid repeat embedding calls, chunk indexing, and their associated startup time and API cost. |
| Separate index warming | `python app.py --warm-index` runs ingestion and index preparation without answering a question. | The expensive work can run before interactive use or from a scheduler, reducing first-user wait time. |
| Embedding controls and metrics | `EMBEDDING_BATCH_SIZE` and `EMBEDDING_MAX_RETRIES` are validated configuration values; ingestion and semantic-index builds emit document, page, chunk, duration, and cache-hit logs. | Operators can tune provider pressure and observe indexing work without changing code. |
| Query routing | A classifier separates document-store questions from document-content questions. | Questions such as “what documents are available?” avoid unnecessary retrieval and embedding work. |
| Query rephrasing | A lightweight chain rewrites content questions into standalone retrieval queries. | Improves retrieval when a chat-style question depends on omitted context or informal phrasing. |
| Focused answer generation | `RetrievalQA` uses the `stuff` chain type with `gpt-4o-mini` at temperature 0. | A concise, repeatable answer is generated from the retrieved context with a low-latency chain shape. |
| Resilience fallbacks | Classifier, rephrasing, and general-answer failures fall back to a safe deterministic behavior. | A transient model error does not turn into a blank or broken user experience. |
| User-facing error boundaries | Streamlit catches initialization and answer failures, logs technical details server-side, and shows a safe recovery message. The CLI returns a nonzero exit code with a concise error. | Users receive actionable feedback without a raw traceback or leaked provider details. |
| Streamlit caching | Chains/models are cached as resources; the document-store summary is cached as data. | Streamlit reruns remain fast because the index and clients are not rebuilt on every interaction. |
| Secrets fallback | `.env` is loaded first; AWS Secrets Manager is used only when `OPENAI_API_KEY` is absent. | Local development and AWS-backed deployments use the same application code. |
| Containerized local infrastructure | Docker Compose starts the app workspace and a single-node OpenSearch service. | Developers have a reproducible environment for hybrid retrieval without installing services on the host. |
| Offline regression tests | Pytest covers path resolution, loaders, routing fallbacks, retrieval wiring, OpenSearch indexing, and secret-handling branches. | Container setup detects application regressions before development begins, without requiring external APIs. |

## Request Flow

### 1. Startup and index construction

![Index build with manifest cache](index-cache.svg)

*Interactive version: [index-cache.html](index-cache.html).*

1. `app.py` loads environment variables from `.env`. When no API key is
   available, it attempts to load secrets from AWS Secrets Manager.
2. Streamlit imports the application helpers and initializes cached resources.
3. `build_qa_chain_directory()` recursively discovers supported files under
   `doc_files`, loads them, and splits their contents into chunks.
4. The selected retrieval mode builds a FAISS index, an OpenSearch index, or
   both. FAISS embedding requests use batches of up to 100 chunks and retry
   transient failures up to six times.
   - The operator selects the mode with the `RETRIEVER_MODE` environment
     variable in the Streamlit application, or `--retriever` for the CLI.
   The default, `hybrid`, is the right baseline when questions may use
   different wording than the source material and exact technical terms
   also matter. `faiss` is useful when semantic similarity alone is enough.
   `opensearch` is useful for exact terms such as identifiers, acronyms,
   and product names, and it avoids embedding API calls.
   - The 100-chunk embedding batch is a throughput and API-rate-limit
     compromise: larger batches reduce request overhead, while this bounded
     size limits token bursts for a heterogeneous document collection. The
     six retry attempts provide resilience to temporary API failures and HTTP
     429 rate limits during initial indexing; they do not mask persistent
     credential, quota, or input errors.
    - Each supported source file is content-hashed into a deterministic
       manifest. When the manifest matches a saved FAISS index in `.langlab`,
       the semantic index is reused instead of re-embedding unchanged chunks.
       The same manifest selects an immutable OpenSearch generation; a matching
       generation skips bulk indexing. On a change, chunks are indexed into a
       new generation and the stable keyword-search alias is switched only after
       that generation is searchable, avoiding partial live indexes.
5. The Streamlit session retains cached chains and a cached document summary.

Run `python app.py --warm-index --retriever hybrid` before interactive use to
perform the same ingestion/index work as a standalone operation. This
front-loads indexing cost before users arrive, and the manifest cache removes
repeat semantic-embedding and keyword-indexing work for unchanged documents.

### 2. General collection questions

For questions about the collection itself, the classifier selects the general
path:

1. The classifier labels the question as `GENERAL`.
2. The general-answer chain receives only the cached document summary.
3. The response is rendered and kept in Streamlit session history.

This path avoids vector search and gives answers grounded in the actual files
that were discovered.

### 3. Document-content questions

For a question about document content, the application takes the retrieval
path:

![Hybrid retrieval request sequence](hybrid-request.svg)

*Interactive version: [hybrid-request.html](hybrid-request.html).*

1. The rephraser turns the input into a standalone search question. If it
   fails or produces no text, the original question is used.
2. The selected retriever returns the top four chunks:
   - `faiss`: semantic similarity only.
   - `opensearch`: keyword/BM25 matching only.
   - `hybrid`: reciprocal-rank fusion of both result sets.
3. `RetrievalQA` sends the retrieved context and question to the answer model.
4. The UI displays the normalized answer; empty answers receive a clear
   fallback message instead of appearing blank.

Together, routing, rephrasing, retrieval, and answer generation reduce
unnecessary model calls while preserving a path for semantically difficult
questions and exact technical terminology.

## Workload Fit and Scaling Boundaries

### Best fit: interactive, curated knowledge bases

The current architecture is well suited to a team, training, research, or
support knowledge base with roughly 10 to 5,000 documents, a modest number of
concurrent interactive users, and content that changes periodically rather
than continuously. This assumes enough memory for the in-process FAISS index,
enough OpenSearch resources for the corresponding chunk count, and use of the
manifest cache or `--warm-index` workflow to avoid re-embedding unchanged
collections. It is especially appropriate when the corpus contains technical
PDFs where users need both semantic questions and exact acronym or identifier
lookup.

FAISS is in-process and the Streamlit cache is process-local. This makes
single-instance startup and query latency straightforward, but it also means
each application process owns its own index and embedding lifecycle. The
single-node OpenSearch container is intended for local development and lab
workloads, not high availability or large-scale search serving.

### Capacity characteristics

| Area | Current behavior | Scaling implication |
| --- | --- | --- |
| Ingestion | Source files are content-hashed; `--warm-index` can run index preparation separately from question answering. | Unchanged collections reuse both retrieval indexes. Changed collections still rebuild as one bounded batch. |
| Semantic index | FAISS runs in the application process and is persisted under `.langlab` when its manifest matches. | Reuse reduces embedding cost and startup time, but memory usage still grows with chunk count and each app process owns its loaded index. |
| Keyword index | OpenSearch keeps manifest-addressed generations and exposes the ready one through a stable alias. | Unchanged collections avoid bulk indexing; an incomplete new generation is never made live. Production still needs multi-node sizing, backups, access control, monitoring, and lifecycle policies. |
| Answering | The model is invoked per classification, rephrasing, and content-answer step as applicable. | User throughput is bounded primarily by model rate limits and latency. Cache resources, not answers, so each new question still consumes model capacity. |
| Web application | Streamlit manages session state in one running process. | Suitable for a small concurrent audience. Multiple replicas need shared session, cache, index, and ingestion coordination. |

### When to evolve the architecture

Move beyond the current topology when any of these become routine:

- The corpus is too large to embed and hold in one application process within
  the acceptable startup window or memory budget.
- Many concurrent users produce model rate-limit pressure or unacceptable
  response latency.
- Documents must be updated continuously, audited, versioned, or indexed
   incrementally at a rate where a full local rebuild on change is no longer
   acceptable.
- The system needs multi-instance availability, tenant isolation, production
  authentication, or disaster recovery.

At that point, keep the request-flow concepts but externalize the state:

1. Run ingestion as a background job that records document versions and
   chunk metadata.
2. Use a persistent managed vector store or a horizontally scalable vector
   index instead of in-process FAISS.
3. Run OpenSearch as a secured, monitored multi-node service when keyword
   search remains useful.
4. Place the chat/API layer behind an application service that supports
   authentication, observability, rate limiting, and shared session state.
5. Add metrics for ingestion duration, chunk counts, retrieval quality,
   model errors, token consumption, and end-to-end latency.

## Quality and Delivery Controls

The development container installs `pytest` from `requirements.txt`. Its
`postCreateCommand` runs secret setup and then the complete test suite. These
tests deliberately mock OpenAI, OpenSearch, and AWS boundaries, so container
creation remains repeatable without network credentials or external services.

This design gives fast feedback for local development while keeping external
integration tests as a separate, explicitly provisioned concern.