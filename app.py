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
    PyPDFDirectoryLoader
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
# LangChain imports for the question-rephrasing preprocessor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Chain type for the RetrievalQA chain 
# -> (in order of speed/cost trade-off: "stuff", "map_reduce", "refine", "map_rerank")
CHAIN_TYPE = "map_rerank"


# Function to build the RetrievalQA chain from a single PDF document
def build_qa_chain_single(pdf_path="AAA-Identity-Management-Security.pdf"):
    """Load a single PDF, build the FAISS vector store, and return a RetrievalQA chain."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    # Split the loaded documents into smaller chunks for embedding
    chunks = splitter.split_documents(docs)

    # FAISS vector store for document embeddings
    # max_retries with backoff and a smaller batch size avoid tokens-per-minute rate limit errors
    embeddings = OpenAIEmbeddings(chunk_size=100, max_retries=6)
    db = FAISS.from_documents(chunks, embeddings)

    # Initialize the language model (LLM) for the RetrievalQA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build and return the RetrievalQA chain using the LLM and FAISS retriever
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=db.as_retriever()
    )


# Function to build the RetrievalQA chain from a directory of PDF documents
def build_qa_chain_directory(pdf_dir="doc_files"):
    """Load all PDFs in a directory, build the FAISS vector store, and return a RetrievalQA chain."""
    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # max_retries with backoff and a smaller batch size avoid tokens-per-minute rate limit errors
    embeddings = OpenAIEmbeddings(chunk_size=100, max_retries=6)
    db = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=db.as_retriever()
    )

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


# Function to summarize the titles/count of PDFs in a document store directory
def get_store_summary(pdf_dir="doc_files"):
    """Build a plain-text summary of the documents found in a directory."""
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    titles = [os.path.splitext(os.path.basename(path))[0] for path in pdf_paths]
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


# Function to parse command-line arguments for questions to ask the PDF document
def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions about the loaded PDF document.")
    parser.add_argument(
        "questions",
        nargs="*",
        default=["What is the title and chapters of this document?"],
        help="One or more questions to ask (default: 'What is the title and chapters of this document?')",
    )
    return parser.parse_args()

# Main entry point for the script
if __name__ == "__main__":
    args = parse_args()
    qa_chain = build_qa_chain_single()
    for question in args.questions:
        result = qa_chain.invoke(question)
        print(f'\n{result}\n')
