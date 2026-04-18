from src.embedding import embed_query
from src.retrieval import retrieve_with_expansion
from src.prompt import build_prompt
from src.llm import get_response


def get_top_candidate(chunks):
    import re

    best = None
    best_votes = -1

    for item in chunks:
        text = item["text"]

        match = re.search(r"Votes\(%\):\s*([\d.]+)%", text)
        if match:
            votes = float(match.group(1))

            if votes > best_votes:
                best_votes = votes
                best = item

    return best


def run_pipeline(query, vector_retriever, keyword_retriever, top_k=5, alpha=0.7):
    retrieved_results = retrieve_with_expansion(
        query=query,
        embed_query_fn=embed_query,
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        k=top_k,
        alpha=alpha
    )

    context_chunks = [item["text"] for item in retrieved_results]
    scores = [item["final_score"] for item in retrieved_results]

    # ⭐ Innovation: find top candidate
    top_candidate = get_top_candidate(retrieved_results)

    # Add structured hint for the LLM
    if top_candidate:
        extra_info = f"""
Top Candidate Based on Votes:
{top_candidate['text']}
"""
        context_chunks.append(extra_info)

    prompt = build_prompt(query, context_chunks)
    answer = get_response(prompt)

    return {
        "retrieved_results": retrieved_results,
        "chunks": context_chunks,
        "scores": scores,
        "prompt": prompt,
        "answer": answer
    }