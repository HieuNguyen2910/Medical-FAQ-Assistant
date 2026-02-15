# ingest.py
import os
import json
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from tqdm import tqdm

# Config
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
DATASET_NAME = "tarudesu/ViHealthQA"

def load_vihealthqa():
    ds = load_dataset(DATASET_NAME, split="train")
    return ds

def simple_chunk_text(text, tokenizer, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    if text is None:
        return []
    toks = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    N = len(toks)
    while i < N:
        j = min(i + chunk_tokens, N)
        slice_ids = toks[i:j]
        chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text.strip())
        if j == N:
            break
        i = j - overlap
    return chunks

def main():
    print("Loading dataset...")
    ds = load_vihealthqa()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)  # tokenizer used for chunking
    docs = []
    print("Chunking and creating documents...")
    for i, row in enumerate(tqdm(ds)):
        # Use 'context' if present, else use question+answer
        context = row.get("context") or ""
        if isinstance(context, list):
            context = "\n".join(context)
        # Add answer as its own document too
        answer = row.get("answer") or ""
        source = row.get("source") if "source" in row else f"{DATASET_NAME}"
        # chunk context
        chunks = simple_chunk_text(context, tokenizer)
        for cid, chunk in enumerate(chunks):
            metadata = {"source": source, "dataset_index": i, "chunk_id": f"{i}_{cid}"}
            docs.append(Document(page_content=chunk, metadata=metadata))
        # also add answer as separate small doc
        if answer:
            docs.append(Document(page_content=answer, metadata={"source": source, "dataset_index": i, "chunk_id": f"{i}_ans"}))
    print(f"Total documents (chunks): {len(docs)}")
    # embeddings
    print("Creating embeddings (this can take time)...")
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # upsert to Chroma
    vectordb = Chroma.from_documents(docs, embedding=emb, persist_directory=CHROMA_DIR)
    vectordb.persist()
    print("Ingestion finished. Chroma persisted at", CHROMA_DIR)

if __name__ == "__main__":
    main()
