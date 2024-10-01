# RAGScholar

RAGScholar is a Q&A chatbot designed specifically for academic papers, aimed at enhancing research collaboration within your group. The entire application is containerized using Docker for easy setup and deployment, and it can be run locally for direct access and usage. By default, it utilizes Llama 3.2 as the large language model (LLM) and stores vectors using ChromaDB.

## Features

- **Academic Focus**: Tailored for answering queries related to research papers.
- **Easy Setup**: Fully dockerized for seamless installation and execution.
- **Local Deployment**: Can be run locally for direct access and usage.
- **Customizable**: Supports changing the LLM model and parameters to fit your needs.

## Setup Instructions

1. **Add Research Papers**
   - Place your research group's papers in the [resources](resources/) directory.

2. **Configure LLM and Parameters**
   - Modify the LLM model and other settings in the [src/constants.py](src/constants.py) file, if required. 

3. **Run the Application**
   - Build and start the dockerized application with the following commands:
     ```bash
     docker-compose build
     docker compose up -d && docker attach application
     ```

# License

This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.