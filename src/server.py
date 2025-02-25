import re
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
    RESOURCES_PDF_PATH, LLM_SYSTEM_PROMPT 
from dataloader import load_all_documents
from termcolor import colored

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
        show_progress=True
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

def runRagLLM(input_msg, ollama_client, vector_store, retriever, messages):

    # Retrieve the documents for the given input_msg
    retrieved_docs = retriever.invoke(input_msg)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # format the prompt with question and context 
    formatted_prompt = f"Question: {input_msg}\n\nContext: {formatted_context}"
    
    # append the user's query + retrieved content to messages
    messages.append({'role': 'user', 'content': formatted_prompt})

    response_stream = ollama_client.chat(model=LLM_MODEL_NAME, messages=messages, stream=True)
 
    return response_stream

if __name__ == "__main__":

    ollama_client = setupOllama()
    vector_store, retriever = setupChromaDB()

    messages = [
        {
            'role': 'system',
            'content': LLM_SYSTEM_PROMPT,
        },
    ]
    while True:
        
        # get user input
        question = input(colored("\nQuestion :: ", "red"))
        
        
        # retrieve the context and generate response stream
        response_stream = runRagLLM(input_msg=question, ollama_client=ollama_client,
                             vector_store=vector_store, retriever=retriever,
                             messages=messages)
        
        # output the llm's response
        response = ""
        # State to track if we are currently inside a think-tag block.
        inside_think = False
        # Buffer to hold any leftover text from previous chunk that wasn't processed.
        buffer = ""
        for chunk in response_stream:
            content = chunk['message']['content']
            response += content
            
            # Prepend any buffered text from previous chunks.
            text = buffer + content
            buffer = ""  # Clear buffer after concatenation
            
            # Process the text in a loop in case there are multiple tags.
            while text:
                if not inside_think:
                    # Look for the next opening tag.
                    open_match = re.search(r'\<think\>', text)
                    if open_match:
                        # Print text before the tag normally.
                        before_tag = text[:open_match.start()]
                        print(before_tag, end='', flush=True)
                        # Print the opening tag in yellow.
                        print(colored('<think>', 'yellow'), end='', flush=True)
                        # Set state to inside think.
                        inside_think = True
                        # Continue processing after the opening tag.
                        text = text[open_match.end():]
                    else:
                        # No opening tag found; print the remaining text normally.
                        print(text, end='', flush=True)
                        text = ""
                else:
                    # We're inside a think block; look for a closing tag.
                    close_match = re.search(r'\</think\>', text)
                    if close_match:
                        # Print text up to the closing tag in yellow.
                        inside_text = text[:close_match.start()]
                        print(colored(inside_text, 'yellow'), end='', flush=True)
                        # Print the closing tag in yellow.
                        print(colored('</think>', 'yellow'), end='', flush=True)
                        # Exit the think block.
                        inside_think = False
                        # Continue processing after the closing tag.
                        text = text[close_match.end():]
                    else:
                        # No closing tag found; print all in yellow and break out.
                        print(colored(text, 'yellow'), end='', flush=True)
                        # Save nothing to the buffer because we've printed all available text.
                        text = ""
            
            # If the chunk ends in the middle of a tag sequence, buffer remains empty here.
            # If needed, you can adjust the logic to store partial tag text in buffer.
            # (This sample assumes tags won't be split in the middle of the tag string itself.)
        
        # Append the model's complete response to the messages.
        messages.append({'role': 'assistant', 'content': response})