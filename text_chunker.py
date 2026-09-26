import semchunk
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

chunker = semchunk.chunkerify(tokenizer, chunk_size=200)

embedding_model = SentenceTransformer(MODEL_NAME)

def chunk_text(text):
    """
    Safely chunk text into smaller pieces based on lines and token counts.
    """
    return chunker(text)

chunks = chunk_text("Your long text goes here...")
for i, chunk in enumerate(chunks):
    print(f"=== Chunk {i + 1} ({len(chunk)} characters) ===")
    print(chunk)
    print("=" * 30 + "\n")

embeddings = embedding_model.encode(chunks)

print(f"Generated {len(chunks)} chunks.")
print(f"Embedding matrix shape: {embeddings.shape}")
