import os 
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

class DocumentLoader:
  
    def __init__(self, file_path, text_splitter):
        self.file_path = file_path
        self.text_splitter = text_splitter
        self.documents = []
        self.chunked_documents = None
        self.__read_documents()
        
    def load_into_database(self, chroma_client, embeddings, collection_name):

        print("Creating embeddings for the chunks")

        # Adding only documents that are not already present in the database 
        # Create a list of unique ids for each document based on the content
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in self.chunked_documents]
        unique_ids = list(set(ids))

        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [doc for doc, id in zip(self.chunked_documents, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

        vectordb = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            ids=unique_ids,
            client=chroma_client,
            collection_name=collection_name,
        )
        print("Done with embeddings creation")

        return vectordb


    def __read_documents(self):

        for file in os.listdir(self.file_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.file_path, file)
                loader = PyPDFLoader(pdf_path)
                self.documents.extend(loader.load())

        print(f"Loaded {len(self.documents)} documents")
        self.chunked_documents = self.text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunked_documents)} chunked documents")
        


