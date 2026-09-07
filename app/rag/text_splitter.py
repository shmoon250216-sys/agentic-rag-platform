def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split long text into overlapping chunks for retrieval."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = end - overlap
    return chunks
