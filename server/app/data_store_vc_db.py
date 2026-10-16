
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    BASE_DIR.parent / "student_management_transport_guide.pdf"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768,
)


def ingest_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # 1. Load and extract PDF text.
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError("No pages found in the PDF.")

    # 2. Split text into smaller chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text extracted. Check whether the PDF is scanned."
        )

    # 3. Store chunks and their embeddings in Pinecone.
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.getenv(
            "PINECONE_INDEX_NAME", "school-help"
        ),
        namespace="school-manual",
    )

    print(f"PDF pages loaded: {len(documents)}")
    print(f"Chunks uploaded: {len(chunks)}")
    print("PDF ingestion completed.")


if __name__ == "__main__":
    ingest_pdf()
