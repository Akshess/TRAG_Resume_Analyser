from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def rerank(query : str, documents: list[str], top_k:int = 3):

    pairs = [(query,doc) for doc in documents]

    scores = model.predict(pairs)
    
    scored_docs = list(zip(documents,scores))

    ranked = sorted(scored_docs, key = lambda x: x[1], reverse = True)

    return [doc for doc,score in ranked[:top_k]]
