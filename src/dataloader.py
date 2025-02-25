import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document


##############################
# PDF Document Loader Class  #
##############################

class PDFDocumentLoader:
    def __init__(self, file_path, text_splitter):
        self.file_path = file_path
        self.text_splitter = text_splitter
        self.documents = []
        self.chunked_documents = None
        self.__read_documents()
        
    def load_documents(self):
        # Returns the chunked documents so that they can be merged with others
        return self.chunked_documents

    def __read_documents(self):
        for file in os.listdir(self.file_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(self.file_path, file)
                loader = PyPDFLoader(pdf_path)
                self.documents.extend(loader.load())
        print(f"Loaded {len(self.documents)} PDF documents")
        self.chunked_documents = self.text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunked_documents)} PDF chunked documents")


#################################
# JSON Document Loader Class    #
#################################

class JSONDocumentLoader:
    def __init__(self, file_path, text_splitter):
        self.file_path = file_path
        self.text_splitter = text_splitter
        self.documents = []
        self.chunked_documents = None
        self.__read_documents()
        
    def load_documents(self):
        # Returns the chunked documents so that they can be merged with others
        return self.chunked_documents

    def __read_documents(self):
        json_files = [file for file in os.listdir(self.file_path) if file.endswith('.json')]
        
        def process_file(file):
            json_path = os.path.join(self.file_path, file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                extracted_text = self.__extract_relevant_text(data)
                metadata = {
                    "title": data.get("title", ""),
                    "filename": file,
                    "identifier": data.get("identifier", "")
                }
                return Document(page_content=extracted_text, metadata=metadata)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_file, file) for file in json_files]
            for future in as_completed(futures):
                doc = future.result()
                if doc is not None:
                    self.documents.append(doc)
                    
        print(f"Loaded {len(self.documents)} JSON documents")
        self.chunked_documents = self.text_splitter.split_documents(self.documents)
        print(f"Created {len(self.chunked_documents)} JSON chunked documents")

    def __extract_relevant_text(self, json_data):
        relevant_text = []
        # Extract key textual fields
        relevant_text.append(json_data.get("title", ""))
        relevant_text.append(json_data.get("notes", ""))
        relevant_text.append(json_data.get("name", ""))
        relevant_text.append(json_data.get("disname", ""))
        relevant_text.append(str(json_data.get("origin", "")))
        relevant_text.append(str(json_data.get("proliferation", "")))
        
        # Extract text from sections
        if "sections" in json_data:
            for section in json_data["sections"]:
                relevant_text.append(section.get("name", ""))
                for prop in section.get("properties", []):
                    relevant_text.append(f"{prop.get('name', '')}: {prop.get('value', '')}")

        return " ".join(filter(None, relevant_text))


#################################
# Combined Loader Function      #
#################################

def load_all_documents(pdf_path, json_path, text_splitter):
    # Instantiate both loaders
    pdf_loader = PDFDocumentLoader(file_path=pdf_path, text_splitter=text_splitter)
    json_loader = JSONDocumentLoader(file_path=json_path, text_splitter=text_splitter)

    # Combine chunked documents from both loaders
    combined_chunked_documents = []
    if pdf_loader.load_documents():
        combined_chunked_documents.extend(pdf_loader.load_documents())
    if json_loader.load_documents():
        combined_chunked_documents.extend(json_loader.load_documents())
    print(f"Total combined chunked documents: {len(combined_chunked_documents)}")
    return combined_chunked_documents
