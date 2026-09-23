import os

import streamlit as st

from app import (
    build_qa_chain_directory,
    build_question_rephraser,
    rephrase_question,
    build_question_classifier,
    is_general_store_question,
    build_general_answer_chain,
    answer_general_question,
    get_store_summary,
)

st.set_page_config(page_title="LangLab Chatbot", page_icon="🤖")
st.title("LangLab Chatbot")


@st.cache_resource
def get_qa_chain():
    # RETRIEVER_MODE selects "faiss" (default), "opensearch", or "hybrid" retrieval
    return build_qa_chain_directory(retriever_mode=os.environ.get("RETRIEVER_MODE", "faiss"))


@st.cache_resource
def get_rephraser():
    return build_question_rephraser()


@st.cache_resource
def get_classifier():
    return build_question_classifier()


@st.cache_resource
def get_general_answer_chain():
    return build_general_answer_chain()


@st.cache_data
def get_cached_store_summary():
    return get_store_summary()


qa_chain = get_qa_chain()
rephraser = get_rephraser()
classifier = get_classifier()
general_answer_chain = get_general_answer_chain()
store_summary = get_cached_store_summary()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_general_store_question(classifier, question):
                answer = answer_general_question(general_answer_chain, question, store_summary)
            else:
                rephrased_question = rephrase_question(rephraser, question)
                result = qa_chain.invoke(rephrased_question)
                answer = result["result"] if isinstance(result, dict) else str(result)
                # map_rerank can return an empty string when no chunk scores confidently
                answer = answer.strip() or "I couldn't find a confident answer to that question in the documents."
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
