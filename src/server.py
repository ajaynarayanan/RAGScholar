import chromadb
from chromadb.config import Settings
from ollama import Client
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import OLLAMA_HOST, OLLAMA_PORT, CHROMADB_HOST, CHROMADB_PORT, RESOURCES_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME
from dataloader import DocumentLoader

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def setupOllama():
    ollama_client = Client(host=f"{OLLAMA_HOST}:{OLLAMA_PORT}")
    ollama_client.pull(LLM_MODEL_NAME)
    ollama_client.pull(EMBEDDING_MODEL_NAME)

    return ollama_client

def setupChromaDB():
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, settings=Settings(allow_reset=True, anonymized_telemetry=False))

    embed_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
    )

    document_loader = DocumentLoader(file_path=RESOURCES_PATH, text_splitter=TEXT_SPLITTER)
    vector_store = document_loader.load_into_database(chroma_client=chroma_client, embeddings=embed_model, collection_name="my_documents")

    # results = vector_store.similarity_search(query="get me anything random",k=1)
    # for doc in results:
    #     print(f"* {doc.page_content} [{doc.metadata}]")

    # RAG Setup
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