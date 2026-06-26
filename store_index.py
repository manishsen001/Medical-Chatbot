from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")
INDEX_NAME = "medical-chatbot"
INDEX_DIMENSION = int(os.getenv("PINECONE_INDEX_DIMENSION", "384"))

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = download_hugging_face_embeddings()

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=INDEX_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

tests_chunk = []
if PDF_PATH:
    documents = load_pdf_file(PDF_PATH)
    minimal_docs = filter_to_minimal_docs(documents)
    tests_chunk = text_split(minimal_docs)

if tests_chunk:
    docsearch = PineconeVectorStore.from_documents(
        documents=tests_chunk,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
else:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
