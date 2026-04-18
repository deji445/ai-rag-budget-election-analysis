def build_prompt(query, context_chunks, max_chunks=3):
    # Limit context size (important for performance + accuracy)
    selected_chunks = context_chunks[:max_chunks]

    context = "\n\n".join(selected_chunks)

    prompt = f"""
You are a factual AI assistant.

Your task is to answer the user's question ONLY using the provided context.

STRICT RULES:
- Do NOT use outside knowledge.
- Do NOT make assumptions.
- If the answer is not clearly found in the context, say: "I don't know".
- Be concise and direct.
- If possible, reference specific details from the context.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt