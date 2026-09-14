# App Initialization Order of Operations

This document describes, in execution order, what happens when the Streamlit
chatbot is launched with:

```bash
streamlit run chat_app.py
```

## 1. Module import: `app.py` is loaded

Because [chat_app.py](../chat_app.py) does `from app import (...)`, Python
fully executes [app.py](../app.py) top-to-bottom before `chat_app.py`
continues. This happens once per process (Streamlit re-imports on file
change, but not on every rerun).

1. **Environment setup**
   - `load_dotenv()` reads key/value pairs from a local `.env` file (if
     present) into `os.environ`. This is where `OPENAI_API_KEY` normally
     comes from in local development.
   - If `OPENAI_API_KEY` is still not present in the environment after
     `load_dotenv()`, `load_secrets()` (from [load_secrets.py](../load_secrets.py))
     is called as a fallback — e.g. to pull the key from AWS Secrets Manager
     or another secret store. This lets the same code run locally (via
     `.env`) and in a deployed environment (via a secrets backend) without
     changes.
2. **Library imports**
   - Document loaders: `PyPDFLoader` (single file) and
     `PyPDFDirectoryLoader` (whole directory).
   - `RecursiveCharacterTextSplitter` for chunking loaded pages into
     embedding-sized pieces.
   - `RetrievalQA` (from `langchain_classic.chains`) — the chain that ties a
     retriever and an LLM together to answer questions.
   - `OpenAIEmbeddings` and `ChatOpenAI` — the embedding model and chat
     model clients, respectively.
   - `FAISS` — in-memory vector store used to index chunk embeddings and
     perform similarity search.
   - `PromptTemplate` / `StrOutputParser` — building blocks for the small
     auxiliary LLM chains (rephraser, classifier, general-answer chain).
   - None of these imports make network calls by themselves; they just
     prepare classes/functions for later use.
3. **Module-level constants are defined** (no API calls yet):
   - `CHAIN_TYPE = "map_rerank"` — controls how `RetrievalQA` combines
     retrieved chunks into a final answer (see chain-type trade-offs below).
   - `REPHRASE_PROMPT` — prompt template used to rewrite a raw user question
     into a clearer, standalone question before retrieval.
   - `CLASSIFY_PROMPT` — prompt template used to label a question as
     `GENERAL` (about the document store itself) or `SPECIFIC` (about
     document content).
   - `GENERAL_ANSWER_PROMPT` — prompt template used to answer `GENERAL`
     questions directly from a store summary, bypassing retrieval entirely.
4. **Function definitions** — none of the following execute yet; they are
   only defined so `chat_app.py` can call them on demand:
   - `build_qa_chain_single(pdf_path=...)` — single-PDF pipeline (used by the
     CLI entry point in `app.py`, not by the web UI).
   - `build_qa_chain_directory(pdf_dir="doc_files")` — directory-of-PDFs
     pipeline (used by the web UI).
   - `build_question_rephraser()` / `rephrase_question(...)`.
   - `get_store_summary(pdf_dir="doc_files")`.
   - `build_question_classifier()` / `is_general_store_question(...)`.
   - `build_general_answer_chain()` / `answer_general_question(...)`.
   - `parse_args()` — CLI argument parsing, only relevant when `app.py` is
     run directly (`python app.py "question"`), not through Streamlit.

## 2. Streamlit page shell

Back in [chat_app.py](../chat_app.py):

- `st.set_page_config(page_title="LangLab Chatbot", page_icon="🤖")` sets the
  browser tab title/icon. Must be the first Streamlit call in the script.
- `st.title("LangLab Chatbot")` renders the page heading.

## 3. Cached resource construction

Streamlit reruns the entire script on every user interaction, so expensive
setup is wrapped in `@st.cache_resource` (for live objects like chains/models)
or `@st.cache_data` (for plain data like strings) to ensure it only runs once
per session and is reused across reruns:

1. `get_qa_chain()` → `build_qa_chain_directory()`:
   - `PyPDFDirectoryLoader("doc_files").load()` reads every PDF under
     `doc_files/` into a list of `Document` objects (one per page).
   - `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
     splits those pages into overlapping ~500-character chunks so each chunk
     fits comfortably within embedding/LLM context limits while preserving
     some cross-chunk continuity via the 50-character overlap.
   - `OpenAIEmbeddings(chunk_size=100, max_retries=6)` embeds the chunks in
     batches of 100 texts per API call, retrying up to 6 times with backoff
     on transient errors (e.g. HTTP 429 rate limits).
   - `FAISS.from_documents(chunks, embeddings)` builds an in-memory vector
     index from the embedded chunks.
   - `ChatOpenAI(model="gpt-4o-mini", temperature=0)` initializes the answer
     generation model with deterministic (temperature 0) output.
   - `RetrievalQA.from_chain_type(llm=llm, chain_type=CHAIN_TYPE,
     retriever=db.as_retriever())` wires the retriever and LLM together.
     With `chain_type="map_rerank"`, the chain asks the LLM to answer using
     each retrieved chunk independently, scores each candidate answer's
     confidence, and returns the highest-scoring one.
2. `get_rephraser()` → `build_question_rephraser()`: builds a
   `REPHRASE_PROMPT | ChatOpenAI | StrOutputParser()` chain that rewrites
   ambiguous or poorly-formed user input into a clear, standalone question
   before it is sent to the retriever — this reduces blank/low-confidence
   answers from `map_rerank`.
3. `get_classifier()` → `build_question_classifier()`: builds a
   `CLASSIFY_PROMPT | ChatOpenAI | StrOutputParser()` chain that labels each
   incoming question `GENERAL` or `SPECIFIC`.
4. `get_general_answer_chain()` → `build_general_answer_chain()`: builds a
   `GENERAL_ANSWER_PROMPT | ChatOpenAI | StrOutputParser()` chain used only
   for `GENERAL` questions.
5. `get_cached_store_summary()` → `get_store_summary("doc_files")`: globs
   `doc_files/**/*.pdf`, strips each path down to a bare filename/title, and
   returns a formatted string like:
   ```
   The document store contains 2 document(s):
   - AAA-Identity-Management-Security
   - Another-Document-Title
   ```
   This step does not call any LLM or embedding API — it's pure filesystem
   inspection — so it's cheap even without caching, but caching still avoids
   redundant disk globbing on every rerun.

All five cached objects/values (`qa_chain`, `rephraser`, `classifier`,
`general_answer_chain`, `store_summary`) are computed once and held in
Streamlit's resource/data cache for the lifetime of the session (or until the
underlying source files change, which invalidates the cache).

## 4. Chat session state

- `st.session_state.messages` is initialized to an empty list the first time
  the app runs for a given browser session, then persisted across reruns via
  Streamlit's session state mechanism.
- Every prior message in `st.session_state.messages` is replayed into the UI
  via `st.chat_message(...)`/`st.markdown(...)` so the conversation appears
  continuous across reruns (Streamlit reruns the whole script on each new
  input, so without this replay step history would disappear).

## 5. Per-question request flow

This block only executes when `st.chat_input(...)` returns a non-empty new
submission:

1. The user's raw question is appended to `st.session_state.messages` and
   rendered immediately.
2. `is_general_store_question(classifier, question)` sends the question
   through the classifier chain. On any exception it defaults to `False`
   (treats the question as specific) rather than failing the request.
3. **If GENERAL:** `answer_general_question(general_answer_chain, question,
   store_summary)` answers directly from the cached store summary, without
   touching the vector store or retrieval chain at all. On failure, it falls
   back to returning the raw `store_summary` text.
4. **If SPECIFIC:**
   - `rephrase_question(rephraser, question)` rewrites the question for
     clarity, falling back to the original question on failure or empty
     output.
   - `qa_chain.invoke(rephrased_question)` runs the full
     retrieve-then-map_rerank pipeline against the FAISS index and returns a
     dict with a `"result"` key (or a bare string, depending on chain
     version).
   - The result is normalized to a string and stripped. If it's empty (which
     `map_rerank` can produce when no retrieved chunk scores confidently), a
     fixed fallback message is shown instead of a blank response.
5. The final answer is rendered via `st.markdown(...)` and appended to
   `st.session_state.messages` so it persists in the visible chat history.

## Chain-type trade-offs (for reference)

`CHAIN_TYPE` in [app.py](../app.py) controls how `RetrievalQA` combines
retrieved chunks, in order of increasing latency/cost:

- `"stuff"` — concatenates all retrieved chunks into one prompt. Fastest,
  but limited by the model's context window.
- `"map_reduce"` — summarizes each chunk independently, then combines the
  summaries. Scales to more/larger documents at the cost of more LLM calls.
- `"refine"` — processes chunks sequentially, refining the answer with each
  new chunk. Sequential, so slower, but can build a more nuanced answer.
- `"map_rerank"` (current default here) — answers independently per chunk
  with a confidence score, then picks the highest-scoring answer. Good when
  the answer likely lives in a single chunk, but can return an empty/blank
  answer when no chunk scores confidently — mitigated here by the
  rephrasing preprocessor and the blank-answer fallback message.
