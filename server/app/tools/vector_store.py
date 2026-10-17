import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768,
)

vector_store = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME", "school-help-dense"),
    embedding=embeddings,
    namespace="school-manual",
)

def print_main_answer(retrieved_docs):
    for doc in retrieved_docs:
        return doc.page_content
    
        # print(doc.page_content)

def retrieve_relevant_data(query: str, k: int = 4) -> list:
    """Fetch the most relevant PDF chunks from Pinecone."""
    print("vc db tool calling ...")

    documents = vector_store.similarity_search(
        query=query,
        k=k,
    )
    result=print_main_answer(documents)
    # print("result fetch ////...",result)
    return documents
