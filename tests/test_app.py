import sys
from pathlib import Path
from unittest.mock import MagicMock

import app
import pytest
from langchain_core.documents import Document


class StubChain:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def invoke(self, _inputs):
        if self.error:
            raise self.error
        return self.response


def test_default_document_path_loads_the_bundled_pdf():
    path = Path(app.DEFAULT_DOCUMENT_PATH)

    assert path.is_file()
    assert path.parent.name == "doc_files"
    assert app.load_document(path)[0].metadata["source"] == str(path)


def test_build_single_chain_uses_default_document_path(monkeypatch):
    loaded_paths = []
    monkeypatch.setattr(app, "load_document", lambda path: loaded_paths.append(path) or ["document"])
    monkeypatch.setattr(app, "build_qa_chain", lambda docs, mode, **_kwargs: (docs, mode))

    assert app.build_qa_chain_single(retriever_mode="opensearch") == (["document"], "opensearch")
    assert loaded_paths == [app.DEFAULT_DOCUMENT_PATH]


def test_build_directory_chain_loads_all_supported_documents(tmp_path, monkeypatch):
    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    (tmp_path / "second.md").write_text("second", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"ignored")
    loaded_paths = []

    def fake_load_document(path):
        loaded_paths.append(path)
        return [Document(page_content=Path(path).name)]

    monkeypatch.setattr(app, "load_document", fake_load_document)
    monkeypatch.setattr(app, "build_qa_chain", lambda docs, mode, **_kwargs: (docs, mode))

    documents, retriever_mode = app.build_qa_chain_directory(str(tmp_path), "opensearch")

    assert loaded_paths == [str(tmp_path / "first.txt"), str(tmp_path / "second.md")]
    assert [document.page_content for document in documents] == ["first.txt", "second.md"]
    assert retriever_mode == "opensearch"


def test_document_discovery_and_summary_handle_supported_nested_files(tmp_path):
    (tmp_path / "report.txt").write_text("top-level", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"not a document")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "guide.MD").write_text("nested", encoding="utf-8")

    assert app.find_documents(str(tmp_path)) == [str(nested / "guide.MD"), str(tmp_path / "report.txt")]
    assert app.get_store_summary(str(tmp_path)) == (
        "The document store contains 2 document(s):\n- guide.MD\n- report.txt"
    )


def test_document_manifest_changes_when_file_content_changes(tmp_path):
    document_path = tmp_path / "document.txt"
    document_path.write_text("first version", encoding="utf-8")

    first_manifest = app.build_document_manifest([str(document_path)], tmp_path)
    document_path.write_text("second version", encoding="utf-8")
    second_manifest = app.build_document_manifest([str(document_path)], tmp_path)

    assert first_manifest["documents"][0]["path"] == "document.txt"
    assert first_manifest != second_manifest


def test_faiss_cache_reuses_matching_manifest_and_rebuilds_on_change(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    manifest = {"version": 1, "documents": [{"path": "document.txt", "sha256": "one"}]}
    database = MagicMock()
    database.as_retriever.return_value = "retriever"
    database.save_local.side_effect = lambda path: Path(path).mkdir(parents=True)
    from_documents = MagicMock(return_value=database)
    monkeypatch.setattr(app, "OpenAIEmbeddings", lambda **_kwargs: "embeddings")
    monkeypatch.setattr(app.FAISS, "from_documents", from_documents)

    assert app.build_retriever([Document(page_content="text")], "faiss", manifest, cache_dir) == "retriever"
    database.save_local.assert_called_once()

    cached_database = MagicMock()
    cached_database.as_retriever.return_value = "cached-retriever"
    monkeypatch.setattr(app.FAISS, "load_local", lambda *_args, **_kwargs: cached_database)

    assert app.build_retriever([Document(page_content="text")], "faiss", manifest, cache_dir) == "cached-retriever"
    assert from_documents.call_count == 1

    changed_manifest = {"version": 1, "documents": [{"path": "document.txt", "sha256": "two"}]}
    assert app.build_retriever([Document(page_content="text")], "faiss", changed_manifest, cache_dir) == "retriever"
    assert from_documents.call_count == 2


def test_text_loader_and_unsupported_file_type(tmp_path):
    text_file = tmp_path / "notes.txt"
    unsupported_file = tmp_path / "archive.zip"
    text_file.write_text("A short note", encoding="utf-8")
    unsupported_file.write_bytes(b"not an archive")

    assert app.load_document(str(text_file))[0].page_content == "A short note"
    with pytest.raises(ValueError, match="Unsupported file type"):
        app.load_document(str(unsupported_file))


def test_load_document_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Document file does not exist"):
        app.load_document(tmp_path / "missing.txt")


def test_document_store_validation_rejects_missing_or_empty_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="Document directory does not exist"):
        app.find_documents(tmp_path / "missing")
    with pytest.raises(ValueError, match="No supported documents"):
        app.build_qa_chain_directory(str(tmp_path))


def test_build_qa_chain_rejects_empty_document_input():
    with pytest.raises(ValueError, match="No documents are available"):
        app.build_qa_chain([])


def test_warm_document_store_builds_retriever_without_creating_a_qa_chain(tmp_path, monkeypatch):
    document_path = tmp_path / "document.txt"
    document_path.write_text("searchable text", encoding="utf-8")
    retriever_inputs = []
    monkeypatch.setattr(
        app,
        "build_retriever",
        lambda chunks, mode, **kwargs: retriever_inputs.append((chunks, mode, kwargs)),
    )

    result = app.warm_document_store(str(tmp_path), "faiss")

    assert result["documents"] == 1
    assert result["chunks"] == 1
    assert retriever_inputs[0][1] == "faiss"
    assert retriever_inputs[0][2]["manifest"]["documents"][0]["path"] == "document.txt"


def test_main_reports_failures_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        app,
        "parse_args",
        lambda: type("Args", (), {"retriever": "hybrid", "questions": [], "warm_index": False})(),
    )
    monkeypatch.setattr(app, "build_qa_chain_single", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("service unavailable")))

    assert app.main() == 1
    assert "Error: service unavailable" in capsys.readouterr().err


def test_question_helpers_fall_back_on_empty_or_error():
    assert app.rephrase_question(StubChain("Clear question"), "Original question") == "Clear question"
    assert app.rephrase_question(StubChain("   "), "Original question") == "Original question"
    assert app.rephrase_question(StubChain(error=RuntimeError()), "Original question") == "Original question"

    summary = "The document store contains 1 document(s):\n- report.pdf"
    assert app.answer_general_question(StubChain("Answer"), "What is available?", summary) == "Answer"
    assert app.answer_general_question(StubChain(""), "What is available?", summary) == summary
    assert app.answer_general_question(StubChain(error=RuntimeError()), "What is available?", summary) == summary
    assert app.is_general_store_question(StubChain("general"), "How many documents?")
    assert not app.is_general_store_question(StubChain("specific"), "What is IAM?")
    assert not app.is_general_store_question(StubChain(error=RuntimeError()), "What is IAM?")


def test_parse_args_accepts_questions_and_retriever(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "What is IAM?", "--retriever", "hybrid"])

    args = app.parse_args()

    assert args.questions == ["What is IAM?"]
    assert args.retriever == "hybrid"


def test_parse_args_defaults_to_hybrid_retrieval(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py"])

    args = app.parse_args()

    assert args.retriever == app.DEFAULT_RETRIEVER_MODE == "hybrid"


def test_invalid_retriever_mode_fails_before_external_services():
    with pytest.raises(ValueError, match="retriever_mode must be one of"):
        app.build_retriever([], "invalid")


def test_faiss_retriever_uses_configured_top_k(monkeypatch):
    embeddings = MagicMock()
    database = MagicMock()
    retriever = MagicMock()
    embedding_options = []
    database.as_retriever.return_value = retriever
    monkeypatch.setattr(
        app,
        "OpenAIEmbeddings",
        lambda **kwargs: embedding_options.append(kwargs) or embeddings,
    )
    monkeypatch.setattr(app.FAISS, "from_documents", lambda chunks, value: database)

    assert app.build_retriever([Document(page_content="text")], "faiss") is retriever
    assert embedding_options == [{
        "chunk_size": app.EMBEDDING_BATCH_SIZE,
        "max_retries": app.EMBEDDING_MAX_RETRIES,
    }]
    database.as_retriever.assert_called_once_with(search_kwargs={"k": app.TOP_K})


def test_invalid_embedding_controls_fail_before_provider_calls(monkeypatch):
    monkeypatch.setattr(app, "EMBEDDING_BATCH_SIZE", 0)

    with pytest.raises(ValueError, match="EMBEDDING_BATCH_SIZE"):
        app.build_retriever([Document(page_content="text")], "faiss")


def test_opensearch_retriever_uses_configured_index(monkeypatch):
    client = MagicMock()
    keyword_retriever = MagicMock()
    monkeypatch.setattr(app, "index_chunks_opensearch", lambda _chunks, **_kwargs: client)
    monkeypatch.setattr(
        app,
        "OpenSearchBM25Retriever",
        lambda **kwargs: keyword_retriever,
    )

    retriever = app.build_retriever([Document(page_content="text")], "opensearch")

    assert retriever is keyword_retriever


def test_hybrid_retriever_combines_faiss_and_opensearch(monkeypatch):
    database = MagicMock()
    faiss_retriever = MagicMock()
    database.as_retriever.return_value = faiss_retriever
    client = MagicMock()
    keyword_retriever = MagicMock()
    captured_arguments = {}
    monkeypatch.setattr(app, "OpenAIEmbeddings", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(app.FAISS, "from_documents", lambda _chunks, _embeddings: database)
    monkeypatch.setattr(app, "index_chunks_opensearch", lambda _chunks, **_kwargs: client)
    monkeypatch.setattr(app, "OpenSearchBM25Retriever", lambda **kwargs: keyword_retriever)
    monkeypatch.setattr(
        app,
        "EnsembleRetriever",
        lambda **kwargs: captured_arguments.update(kwargs) or "hybrid-retriever",
    )

    retriever = app.build_retriever([Document(page_content="text")], "hybrid")

    assert retriever == "hybrid-retriever"
    assert captured_arguments == {
        "retrievers": [faiss_retriever, keyword_retriever],
        "weights": app.HYBRID_WEIGHTS,
    }


def test_build_qa_chain_uses_configured_splitter_and_retrieval_options(monkeypatch):
    splitter = MagicMock()
    chunks = [Document(page_content="chunk")]
    splitter.split_documents.return_value = chunks
    llm = MagicMock()
    retriever = MagicMock()
    chain = MagicMock()
    captured_arguments = {}
    monkeypatch.setattr(app, "RecursiveCharacterTextSplitter", lambda **kwargs: splitter)
    monkeypatch.setattr(app, "ChatOpenAI", lambda **kwargs: llm)
    monkeypatch.setattr(app, "build_retriever", lambda value, mode, *_args: retriever)
    monkeypatch.setattr(
        app.RetrievalQA,
        "from_chain_type",
        lambda **kwargs: captured_arguments.update(kwargs) or chain,
    )

    assert app.build_qa_chain([Document(page_content="source")], "faiss") is chain
    splitter.split_documents.assert_called_once()
    assert splitter.split_documents.call_args.args[0][0].page_content == "source"
    assert captured_arguments == {
        "llm": llm,
        "chain_type": app.CHAIN_TYPE,
        "retriever": retriever,
        "return_source_documents": True,
    }


def test_opensearch_indexing_reuses_matching_generation_and_promotes_alias(monkeypatch):
    class FakeIndices:
        def __init__(self):
            self.calls = []
            self.existing = set()

        def exists(self, index):
            self.calls.append(("exists", index))
            return index in self.existing

        def create(self, index, body):
            self.calls.append(("create", index, body))
            self.existing.add(index)

        def refresh(self, index):
            self.calls.append(("refresh", index))

        def update_aliases(self, body):
            self.calls.append(("update_aliases", body))

    class FakeClient:
        def __init__(self):
            self.indices = FakeIndices()

    client = FakeClient()
    indexed_actions = []
    monkeypatch.setattr(app, "OpenSearch", lambda _url: client)
    monkeypatch.setattr(app.helpers, "bulk", lambda _client, actions: indexed_actions.extend(actions))
    chunks = [Document(page_content="chunk text", metadata={"source": "report.pdf", "page": 1})]
    manifest = {"version": 1, "documents": [{"path": "report.pdf", "sha256": "one"}]}

    assert app.index_chunks_opensearch(chunks, "test-index", manifest) is client
    generation_name = app._opensearch_generation_name("test-index", manifest)
    assert [call[0] for call in client.indices.calls] == ["exists", "create", "refresh", "update_aliases"]
    assert indexed_actions == [{"_index": generation_name, "text": "chunk text", "metadata": chunks[0].metadata}]
    assert client.indices.calls[-1][1]["actions"][-1] == {
        "add": {"index": generation_name, "alias": "test-index"}
    }

    assert app.index_chunks_opensearch(chunks, "test-index", manifest) is client
    assert [call[0] for call in client.indices.calls] == [
        "exists", "create", "refresh", "update_aliases", "exists", "update_aliases"
    ]
    assert len(indexed_actions) == 1