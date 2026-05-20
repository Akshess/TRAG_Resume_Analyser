def chunk_text(text, chunk_size = 500 , overlap = 0):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        print(end)

        chunk = text[start:end]
        chunks.append(chunk)
        print(chunks)
        start = end - overlap
    
    return chunks
