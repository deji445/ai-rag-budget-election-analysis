import faiss
import numpy as np
import re


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def extract_vote_percentage(text):
    match = re.search(r"Votes\(%\):\s*([\d.]+)%", str(text))
    if match:
        return float(match.group(1))
    return 0.0


def extract_years(text):
    return re.findall(r"\b(19\d{2}|20\d{2})\b", str(text))


def contains_exact_phrase(query, text):
    return str(query).lower() in str(text).lower()


def get_region_from_query(query):
    known_regions = [
        "savannah region",
        "northern region",
        "north east region",
        "greater accra region",
        "ashanti region",
        "western region",
        "western north region",
        "central region",
        "eastern region",
        "volta region",
        "oti region",
        "upper east region",
        "upper west region",
        "ahafo region",
        "bono region",
        "bono east region"
    ]

    q = query.lower()
    for region in known_regions:
        if region in q:
            return region
    return None


def is_winner_query(query):
    winner_terms = [
        "who won",
        "winner",
        "victor",
        "leading candidate",
        "won in",
        "highest votes",
        "highest vote",
        "top candidate"
    ]
    q = query.lower()
    return any(term in q for term in winner_terms)


class VectorRetriever:
    def __init__(self, embeddings, chunks):
        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype="float32")

        if len(self.embeddings.shape) != 2 or self.embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=5):
        query_embedding = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "chunk_id": idx,
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "vector_score": 1 / (1 + float(dist))
            })

        return results


class KeywordRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.chunk_tokens = [tokenize(chunk["text"]) for chunk in chunks]

    def search(self, query, k=5):
        query_tokens = set(tokenize(query))
        results = []

        if not query_tokens:
            return results

        query_lower = query.lower()
        query_years = extract_years(query)
        query_region = get_region_from_query(query)

        for i, tokens in enumerate(self.chunk_tokens):
            chunk_text = self.chunks[i]["text"]
            chunk_lower = chunk_text.lower()
            token_set = set(tokens)

            overlap = len(query_tokens.intersection(token_set))
            score = overlap / len(query_tokens)

            # Exact phrase boost
            if contains_exact_phrase(query_lower, chunk_lower):
                score += 0.35

            # Region boost
            if query_region and query_region in chunk_lower:
                score += 0.25

            # Year boost
            if query_years:
                chunk_years = extract_years(chunk_text)
                if any(year in chunk_years for year in query_years):
                    score += 0.20

            if score > 0:
                results.append({
                    "chunk_id": i,
                    "text": chunk_text,
                    "source": self.chunks[i]["source"],
                    "keyword_score": score
                })

        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return results[:k]


def hybrid_search(query, query_embedding, vector_retriever, keyword_retriever, k=5, alpha=0.7):
    candidate_k = max(k * 3, 15)

    vector_results = vector_retriever.search(query_embedding, k=candidate_k)
    keyword_results = keyword_retriever.search(query, k=candidate_k)

    combined = {}
    query_lower = query.lower()
    query_years = extract_years(query)
    query_region = get_region_from_query(query)
    winner_query = is_winner_query(query)

    for item in vector_results:
        cid = item["chunk_id"]
        combined[cid] = {
            "chunk_id": cid,
            "text": item["text"],
            "source": item["source"],
            "vector_score": item["vector_score"],
            "keyword_score": 0.0
        }

    for item in keyword_results:
        cid = item["chunk_id"]

        if cid not in combined:
            combined[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "source": item["source"],
                "vector_score": 0.0,
                "keyword_score": item["keyword_score"]
            }
        else:
            combined[cid]["keyword_score"] = item["keyword_score"]

    final_results = []
    for item in combined.values():
        text_lower = item["text"].lower()
        vote_score = extract_vote_percentage(item["text"]) / 100.0

        bonus = 0.0

        # Exact query phrase bonus
        if contains_exact_phrase(query_lower, text_lower):
            bonus += 0.20

        # Region bonus
        if query_region and query_region in text_lower:
            bonus += 0.20

        # Year bonus
        if query_years:
            chunk_years = extract_years(item["text"])
            if any(year in chunk_years for year in query_years):
                bonus += 0.15

        # Winner-intent bonus
        if winner_query:
            bonus += 0.40 * vote_score

        item["vote_score"] = vote_score
        item["final_score"] = (
            alpha * item["vector_score"]
            + (1 - alpha) * item["keyword_score"]
            + bonus
        )

        final_results.append(item)

    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results[:k]


def expand_query(query):
    expansions = {
        "winner": ["won", "victor", "leading candidate", "highest votes"],
        "won": ["winner", "victor", "highest votes"],
        "budget": ["fiscal policy", "economic policy", "financial plan"],
        "election": ["results", "poll", "vote"],
        "tax": ["levy", "revenue", "taxation"],
        "employment": ["jobs", "job creation", "labour"]
    }

    expanded_queries = [query]
    q = query.lower()

    for word, synonyms in expansions.items():
        if word in q:
            for synonym in synonyms:
                expanded_queries.append(f"{query} {synonym}")

    return list(dict.fromkeys(expanded_queries))


def retrieve_with_expansion(query, embed_query_fn, vector_retriever, keyword_retriever, k=5, alpha=0.7):
    expanded_queries = expand_query(query)
    all_results = {}

    for expanded_query in expanded_queries:
        query_embedding = embed_query_fn(expanded_query)
        results = hybrid_search(
            expanded_query,
            query_embedding,
            vector_retriever,
            keyword_retriever,
            k=k,
            alpha=alpha
        )

        for item in results:
            cid = item["chunk_id"]
            if cid not in all_results or item["final_score"] > all_results[cid]["final_score"]:
                all_results[cid] = item

    merged_results = list(all_results.values())
    merged_results.sort(key=lambda x: x["final_score"], reverse=True)
    return merged_results[:k]