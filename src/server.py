import uuid
import chromadb
from chromadb.config import Settings
from ollama import Client

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import OLLAMA_HOST, OLLAMA_PORT, \
    CHROMADB_HOST, CHROMADB_PORT, LLM_MODEL_NAME, \
    EMBEDDING_MODEL_NAME, RESOURCES_JSON_PATH, \
    RESOURCES_PDF_PATH
from dataloader import load_all_documents

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def setupOllama():
    ollama_client = Client(host=f"{OLLAMA_HOST}:{OLLAMA_PORT}")
    ollama_client.pull(LLM_MODEL_NAME)
    ollama_client.pull(EMBEDDING_MODEL_NAME)

    return ollama_client


#################################
# Setup ChromaDB Function       #
#################################

def setupChromaDB():
    # Initialize ChromaDB client
    chroma_client = chromadb.HttpClient(
        host=CHROMADB_HOST, 
        port=CHROMADB_PORT, 
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )

    # Initialize embeddings (using OllamaEmbeddings as per your current setup)
    embed_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
    )

    # Load and combine documents from PDFs and JSONs
    combined_chunked_documents = load_all_documents(
        pdf_path=RESOURCES_PDF_PATH,
        json_path=RESOURCES_JSON_PATH,
        text_splitter=TEXT_SPLITTER
    )

    # Deduplicate documents using unique IDs based on page_content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in combined_chunked_documents]
    unique_ids = list(set(ids))
    seen_ids = set()
    unique_docs = [
        doc for doc, id in zip(combined_chunked_documents, ids)
        if id not in seen_ids and (seen_ids.add(id) or True)
    ]

    # Load all documents into one ChromaDB collection
    vector_store = Chroma.from_documents(
        documents=unique_docs,
        embedding=embed_model,
        ids=unique_ids,
        client=chroma_client,
        collection_name="my_documents"
    )
    print("Done with embeddings creation and combined loading into ChromaDB")

    # Setup retriever for RAG
    retriever = vector_store.as_retriever()

    return vector_store, retriever

def runRagLLM(input_msg, ollama_client, retriever):

    # Retrieve the documents for the given input_msg
    retrieved_docs = retriever.invoke(input_msg)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # format the prompt with question and context 
    formatted_prompt = f"Question: {input_msg}\n\nContext: {formatted_context}"
    
    response = ollama_client.chat(model=LLM_MODEL_NAME, messages=[{'role': 'user', 'content': formatted_prompt}])
    response = response['message']['content']

    return response 

if __name__ == "__main__":

    ollama_client = setupOllama()
    vector_store, retriever = setupChromaDB()

    while True:
        question = input("Question :: ")
        response = runRagLLM(input_msg=question, ollama_client=ollama_client, retriever=retriever)
        print("Response ::: ", response)