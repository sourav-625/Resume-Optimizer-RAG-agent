from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_query(queries):
    """
    Embed a query using the SentenceTransformer.

    Args:
        queries (list): A list of queries to embed.

    Returns:
        list: A list of embedding vectors.
    """
    return embeddings_model.encode(queries).tolist()

def create_vector_database(chunks, database_name="my_collection"):
    """
    Create a vector database using Qdrant.

    Args:
        chunks (list): A list of document chunks.
        database_name (str): The name of the database collection.

    Returns:
        QdrantClient: The Qdrant client instance.
        database_name (str): The name of the database collection.
    """
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name=database_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    points = []
    for i, chunk in enumerate(chunks):
        embedding = embed_query(chunk)
        points.append(PointStruct(id=i, vector=embedding, payload={"text": chunk}))
    client.upsert(
        collection_name=database_name,
        points=points
    )
    return client, database_name

async def query_vector_database(client, collection, query, top_k=5):
    """
    Query the vector database for similar chunks.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection (str): The name of the collection to query.
        query (str): The query to search for.
        top_k (int): The number of top results to return.
    """
    embedding = embed_query([query])
    results = client.search(
        collection_name=collection,
        query_vector=embedding[0],
        limit=top_k
    )
    return [result.payload["text"] for result in results]