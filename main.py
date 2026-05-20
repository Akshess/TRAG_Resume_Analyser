from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from utils.pdf_loader import load_pdf
from utils.chunking import chunk_text
from utils.embedding import embed
from utils.vectordb import add_chunks, query_vector, reset_user_resume, has_resume
from utils.reranker import rerank
from utils.llm_service import score_resume_against_job
import uvicorn
import os

app = FastAPI(title="Resume Analyser")

os.makedirs("data", exist_ok=True)


class JobDescription(BaseModel):
    user_id: str
    job_description: str


@app.post("/upload_resume")
async def upload_resume(user_id: str, file: UploadFile):
    pdf_bytes = await file.read()
    path = f"data/{user_id}_{file.filename}"

    with open(path, "wb") as f:
        f.write(pdf_bytes)

    text = load_pdf(path)
    chunks = chunk_text(text)
    vectors = embed(chunks)

    reset_user_resume(user_id)
    add_chunks(user_id, chunks, vectors)

    return {
        "status": "Resume indexed",
        "user_id": user_id,
        "total_chunks": len(chunks),
    }


@app.post("/analyse_job")
async def analyse_job(payload: JobDescription):
    user_id = payload.user_id
    jd = payload.job_description

    if not has_resume(user_id):
        raise HTTPException(status_code=404, detail="No resume found for this user. Upload one first.")

    query_vec = embed([jd])[0]
    retrieved = query_vector(user_id, query_vec, top_k=8)
    retrieved_docs = retrieved["documents"][0]

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No resume content retrieved.")

    best_docs = rerank(jd, retrieved_docs, top_k=5)
    resume_context = "\n\n".join(best_docs)

    result = score_resume_against_job(jd, resume_context)

    return {
        "user_id": user_id,
        "score": result.get("score"),
        "verdict": result.get("verdict"),
        "matched_strengths": result.get("matched_strengths", []),
        "gaps": result.get("gaps", []),
        "summary": result.get("summary"),
        "resume_context_used": resume_context,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
