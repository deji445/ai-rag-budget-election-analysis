from sentence_transformers import SentenceTransformer
import streamlit as st


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(chunks):
    model = get_model()
    return model.encode(chunks, show_progress_bar=False)


def embed_query(query):
    model = get_model()
    return model.encode([query], show_progress_bar=False)[0]