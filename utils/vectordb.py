import chromadb

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("resumes")


def add_chunks(user_id, chunks, vectors):
    ids = [f"{user_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"user_id": user_id} for _ in chunks]
    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas,
    )


def query_vector(user_id, vector, top_k=5):
    return collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        where={"user_id": user_id},
    )


def reset_user_resume(user_id):
    existing = collection.get(where={"user_id": user_id})
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])


def has_resume(user_id):
    existing = collection.get(where={"user_id": user_id}, limit=1)
    return bool(existing and existing.get("ids"))
