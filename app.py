import streamlit as st

from src.data_loader import load_csv, load_pdf
from src.chunking import chunk_text
from src.embedding import embed_text
from src.retrieval import VectorRetriever, KeywordRetriever
from src.pipeline import run_pipeline


st.set_page_config(page_title="Academic City AI Assistant")
st.title("Academic City AI Assistant")


@st.cache_data
def load_all_data():
    csv_df = load_csv("data/Ghana_Election_Result.csv")
    pdf_text = load_pdf("data/2025-Budget-Statement-and-Economic-Policy_v4.pdf")
    return csv_df, pdf_text


@st.cache_resource
def build_retrievers():
    csv_df, pdf_text = load_all_data()

    # Convert CSV rows into text
    csv_chunks = []
    for i, row in csv_df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in csv_df.columns])
        csv_chunks.append({
            "chunk_id": i,
            "text": row_text,
            "source": "Ghana_Election_Result.csv"
        })

    # Chunk PDF text
    pdf_text_chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)
    pdf_chunks = []
    for i, chunk in enumerate(pdf_text_chunks):
        pdf_chunks.append({
            "chunk_id": len(csv_chunks) + i,
            "text": chunk,
            "source": "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
        })

    # Combine all chunks
    all_chunks = csv_chunks + pdf_chunks

    # Create embeddings once
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_text(texts)

    # Initialize retrievers once
    vector_retriever = VectorRetriever(embeddings, all_chunks)
    keyword_retriever = KeywordRetriever(all_chunks)

    return vector_retriever, keyword_retriever


vector_retriever, keyword_retriever = build_retrievers()

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching and generating answer..."):
        result = run_pipeline(
            query=query,
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            top_k=3,
            alpha=0.7
        )

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved Chunks")
    for item in result["retrieved_results"]:
        st.write(f"**Source:** {item['source']}")
        st.write(f"**Final Score:** {item['final_score']:.4f}")
        st.write(item["text"])
        st.markdown("---")

    st.subheader("Final Prompt")
    st.code(result["prompt"])