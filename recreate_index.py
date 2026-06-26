import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "medical-chatbot"

# Delete old index
if pc.has_index(INDEX_NAME):
    print(f"Deleting index: {INDEX_NAME}")
    pc.delete_index(INDEX_NAME)

# Create new index
print("Creating new index with dimension 384...")

pc.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print("Index created successfully!")
print(pc.describe_index(INDEX_NAME))