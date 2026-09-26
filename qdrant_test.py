# creating embeddings using sentence-transformers
import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(texts):
  embeddings = model.encode(texts)
  return embeddings.tolist()

texts = [
    "Cat is a mammal.",
    "Dog is also a mammal",
    "Birds are not mammals.",
    "Eagle is a bird."
]

embeddings = create_embeddings(texts)

for e in embeddings:
  print(e)
print(len(embeddings[0]))

# qdrant vector database
qdrant = QdrantClient(path="./qdrant_local_data")

dimensions = len(embeddings[0])
print(dimensions)

qdrant.create_collection(
    collection_name = "animals",
    vectors_config = VectorParams(
        size = dimensions,
        distance = Distance.COSINE
    )
)

for i, (text, embedding) in enumerate(zip(texts, embeddings)):
  qdrant.upsert(
      collection_name="animals",
      points=[
          PointStruct(
              id = i,
              vector = embedding,
              payload = {
                  "text": text
              }
          )
      ]
  )

async def search_query(query: str) -> str:

    # Embed user's query
    response = model.encode([query])

    query_vector = response[0]

    # Search Qdrant
    results = qdrant.query_points(
        collection_name="animals",
        query=query_vector,
        limit=2
    )

    # Extract text
    chunks = []

    for result in results.points:
        chunks.append(result.payload["text"])

    return "  ".join(chunks)

async def main():
    query = "What is a mammal?"
    results = await search_query(query)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())