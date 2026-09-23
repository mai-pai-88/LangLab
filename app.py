import argparse
import glob
import os
from dotenv import load_dotenv
from load_secrets import load_secrets

# Load environment variables from .env or Secrets Manager
load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    load_secrets()

# LangChain imports for environment setup and secret loading
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
# LangChain imports for text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# LangChain imports for document loading, text splitting, and QA chain construction
from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)
# LangChain import for vector store (FAISS)
from langchain_community.vectorstores import FAISS
# LangChain imports for the OpenSearch keyword retriever and hybrid (ensemble) retrieval
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from opensearchpy import OpenSearch, helpers
# LangChain imports for the question-rephrasing preprocessor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Chain type for the RetrievalQA chain 
# -> (in order of speed/cost trade-off: "stuff", "map_reduce", "refine", "map_rerank")
CHAIN_TYPE = "stuff"

# Retriever modes: "faiss" (semantic), "opensearch" (BM25 keyword), "hybrid" (both, merged)
RETRIEVER_MODES = ("faiss", "opensearch", "hybrid")
# Number of chunks each retriever returns
TOP_K = 4
# Weights for [FAISS, OpenSearch] results when merging with Reciprocal Rank Fusion
HYBRID_WEIGHTS = [0.5, 0.5]

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
OPENSEARCH_INDEX = "langlab-chunks"

# File extensions loaded as UTF-8 text (plain text, markup, config, data, and source files)
TEXT_EXTENSIONS = (
    ".txt", ".text", ".log", ".csv", ".tsv", ".rst", ".ini", ".cfg", ".conf",
    ".yaml", ".yml", ".toml", ".xml", ".json", ".py", ".md", ".markdown",
)
# All file extensions the document store can load
SUPPORTED_EXTENSIONS = (".pdf", ".docx") + TEXT_EXTENSIONS


# Retriever that runs a BM25 keyword (match) query against an OpenSearch index
class OpenSearchBM25Retriever(BaseRetriever):
    """Return the top-k chunks from OpenSearch ranked by BM25 keyword relevance."""
    client: OpenSearch
    index_name: str
    k: int = TOP_K

    def _get_relevant_documents(self, query, *, run_manager=None):
        response = self.client.search(
            index=self.index_name,
            body={"size": self.k, "query": {"match": {"text": query}}},
        )
        return [
            Document(page_content=hit["_source"]["text"], metadata=hit["_source"]["metadata"])
            for hit in response["hits"]["hits"]
        ]


# Function to (re)create the OpenSearch index and load the document chunks into it
def index_chunks_opensearch(chunks, index_name=OPENSEARCH_INDEX):
    """Rebuild the OpenSearch index from the chunks and return a connected client."""
    client = OpenSearch(OPENSEARCH_URL)
    # Recreate the index so it always matches the chunks held in FAISS
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    # "dynamic": False keeps all metadata in _source but only indexes the fields mapped here
    client.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {
                    "dynamic": False,
                    "properties": {
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                    },
                },
            }
        }
    })
    helpers.bulk(client, (
        {"_index": index_name, "text": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ))
    # Make the new documents searchable immediately
    client.indices.refresh(index=index_name)
    return client


# Function to build the retriever for the chosen mode from the document chunks
def build_retriever(chunks, retriever_mode="faiss"):
    """Build a FAISS, OpenSearch BM25, or hybrid (FAISS + BM25) retriever over the chunks."""
    if retriever_mode not in RETRIEVER_MODES:
        raise ValueError(f"retriever_mode must be one of {RETRIEVER_MODES}, got {retriever_mode!r}")

    faiss_retriever = None
    if retriever_mode in ("faiss", "hybrid"):
        # FAISS vector store for document embeddings
        # max_retries with backoff and a smaller batch size avoid tokens-per-minute rate limit errors
        embeddings = OpenAIEmbeddings(chunk_size=100, max_retries=6)
        db = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = db.as_retriever(search_kwargs={"k": TOP_K})
        if retriever_mode == "faiss":
            return faiss_retriever

    # OpenSearch keyword index (no embeddings needed, so no OpenAI cost)
    client = index_chunks_opensearch(chunks)
    keyword_retriever = OpenSearchBM25Retriever(client=client, index_name=OPENSEARCH_INDEX)
    if retriever_mode == "opensearch":
        return keyword_retriever

    # Merge semantic and keyword results using weighted Reciprocal Rank Fusion
    return EnsembleRetriever(
        retrievers=[faiss_retriever, keyword_retriever],
        weights=HYBRID_WEIGHTS,
    )


# Function to split documents and build the RetrievalQA chain with the chosen retriever
def build_qa_chain(docs, retriever_mode="faiss"):
    """Split the documents into chunks and return a RetrievalQA chain over them."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # Split the loaded documents into smaller chunks for embedding/indexing
    chunks = splitter.split_documents(docs)

    # Initialize the language model (LLM) for the RetrievalQA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build and return the RetrievalQA chain using the LLM and selected retriever
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=build_retriever(chunks, retriever_mode),
        return_source_documents=True,
    )


# Function to load a single document with the loader that matches its file extension
def load_document(path):
    """Load a PDF, DOCX, or text-based file and return its LangChain documents."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext in TEXT_EXTENSIONS:
        return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
    raise ValueError(f"Unsupported file type {ext!r}; expected one of {SUPPORTED_EXTENSIONS}")


# Function to list the supported document files in a directory (recursively)
def find_documents(doc_dir="doc_files"):
    """Return the sorted paths of all supported, non-hidden files under a directory."""
    paths = glob.glob(os.path.join(doc_dir, "**", "*"), recursive=True)
    return sorted(
        path for path in paths
        if os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTENSIONS)
    )


# Function to build the RetrievalQA chain from a single document
def build_qa_chain_single(doc_path="AAA-Identity-Management-Security.pdf", retriever_mode="faiss"):
    """Load a single document and return a RetrievalQA chain using the chosen retriever."""
    docs = load_document(doc_path)
    return build_qa_chain(docs, retriever_mode)


# Function to build the RetrievalQA chain from a directory of documents
def build_qa_chain_directory(doc_dir="doc_files", retriever_mode="faiss"):
    """Load all supported documents in a directory and return a RetrievalQA chain."""
    docs = [doc for path in find_documents(doc_dir) for doc in load_document(path)]
    return build_qa_chain(docs, retriever_mode)

# Prompt used to rephrase a raw question into a clearer, standalone question
REPHRASE_PROMPT = PromptTemplate.from_template(
    "Rewrite the following question as a single, clear, standalone question "
    "suitable for searching a document. Preserve the original intent and do "
    "not answer it. Return only the rewritten question, with no extra text.\n\n"
    "Question: {question}\n"
    "Rewritten question:"
)


# Function to build a lightweight chain that rephrases questions before retrieval
def build_question_rephraser():
    """Build a chain that rewrites a raw question into a clearer, standalone question."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return REPHRASE_PROMPT | llm | StrOutputParser()


# Function to rephrase a question, falling back to the original on failure or empty output
def rephrase_question(rephraser, question):
    """Rephrase a question using the given rephraser chain, falling back to the original."""
    try:
        rephrased = rephraser.invoke({"question": question}).strip()
    except Exception:
        return question
    return rephrased or question


# Function to summarize the titles/count of documents in a document store directory
def get_store_summary(doc_dir="doc_files"):
    """Build a plain-text summary of the documents found in a directory."""
    titles = [os.path.basename(path) for path in find_documents(doc_dir)]
    doc_list = "\n".join(f"- {title}" for title in titles) or "(no documents found)"
    return f"The document store contains {len(titles)} document(s):\n{doc_list}"


# Prompt used to classify a question as being about the store itself vs. document content
CLASSIFY_PROMPT = PromptTemplate.from_template(
    "Decide whether the question below is a general question about the "
    "document store itself (e.g. how many documents/titles it has, or an "
    "overview of what's in the collection) rather than a specific question "
    "about the content within the documents. Respond with exactly one word: "
    "GENERAL or SPECIFIC.\n\n"
    "Question: {question}\n"
    "Answer:"
)


# Function to build a lightweight chain that classifies questions as GENERAL or SPECIFIC
def build_question_classifier():
    """Build a chain that classifies a question as GENERAL (about the store) or SPECIFIC."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return CLASSIFY_PROMPT | llm | StrOutputParser()


# Function to determine if a question is a general, store-wide question
def is_general_store_question(classifier, question):
    """Return True if the classifier labels the question as GENERAL, defaulting to False."""
    try:
        label = classifier.invoke({"question": question}).strip().upper()
    except Exception:
        return False
    return label.startswith("GENERAL")


# Prompt used to answer a general, store-wide question using the store summary only
GENERAL_ANSWER_PROMPT = PromptTemplate.from_template(
    "Answer the question using only the document store summary below. Do "
    "not invent documents that aren't listed.\n\n"
    "{store_summary}\n\n"
    "Question: {question}\n"
    "Answer:"
)


# Function to build a chain that answers general, store-wide questions
def build_general_answer_chain():
    """Build a chain that answers general questions about the document store."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return GENERAL_ANSWER_PROMPT | llm | StrOutputParser()


# Function to answer a general, store-wide question, falling back to the raw summary on failure
def answer_general_question(chain, question, store_summary):
    """Answer a general question about the document store, falling back to the summary."""
    try:
        answer = chain.invoke({"question": question, "store_summary": store_summary}).strip()
    except Exception:
        return store_summary
    return answer or store_summary


# Function to parse command-line arguments for questions to ask the document
def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions about the loaded document.")
    parser.add_argument(
        "questions",
        nargs="*",
        default=["What is the title and chapters of this document?"],
        help="One or more questions to ask (default: 'What is the title and chapters of this document?')",
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVER_MODES,
        default="faiss",
        help="Retrieval method: faiss (semantic), opensearch (BM25 keyword), or hybrid (both) (default: faiss)",
    )
    return parser.parse_args()

# Main entry point for the script
if __name__ == "__main__":
    args = parse_args()
    qa_chain = build_qa_chain_single(retriever_mode=args.retriever)
    for question in args.questions:
        result = qa_chain.invoke(question)
        print(f'\n[{args.retriever}] {result["query"]}\n{result["result"]}\n')
        # Show which chunks were retrieved so the retriever modes can be compared
        print("Sources:")
        for doc in result["source_documents"]:
            snippet = " ".join(doc.page_content.split())[:100]
            print(f'  - {doc.metadata.get("source")} p.{doc.metadata.get("page")}: {snippet}...')
        print()
