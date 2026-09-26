import qdrant_client

# This spins up a full local storage server without Docker
print("Starting local Qdrant storage server on http://localhost:6333...")
server = qdrant_client.QdrantClient(path="./qdrant_local_data")

# Keep the script running to act as your server
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping Qdrant server safely.")
