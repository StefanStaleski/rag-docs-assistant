import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Docs Assistant", page_icon="🤖", layout="centered")
st.title("🤖 ML Docs Assistant")
st.caption("Ask anything about HuggingFace and LangChain documentation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s}]({s})")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            response = requests.post(f"{API_URL}/query", json={"question": prompt})
            data = response.json()
            answer = data["answer"]
            sources = [s for s in data["sources"] if s]

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- [{s}]({s})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })