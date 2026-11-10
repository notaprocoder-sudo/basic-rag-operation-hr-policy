import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS


load_dotenv()

def load_documents(doc_path="data"):


    # Check if path exists to load documents from.
    if not os.path.exists("data"):
        raise FileNotFoundError("Directory does not exist.")

    # Initiate the loader
    loader = DirectoryLoader(doc_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    # Load Documents
    documents = loader.load()

    # Check if Documents is empty or not.
    if len(documents) == 0:
        raise FileNotFoundError(f"There are no .txt files in the {doc_path} directory.")

    return documents


def doc_chunking(document, chunk_size=800, overlap=0):
    # Initiate document splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    ) 

    # Start splitting documents
    chunks = text_splitter.split_documents(document)

    # Check if chunks are empty
    if len(chunks) == 0:
        raise FileNotFoundError("Splitted Chunks are empty")

    return chunks

def vector_storing(chunks):
    # Initiate the embedding and vector storing of the chunks
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(chunks, embedding)

    stored_vectors = vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={"k":3}
            )

    return stored_vectors


def rag_resp():

    document = load_documents(doc_path="data")
    chunks = doc_chunking(document)
    retriever = vector_storing(chunks)
    model=ChatOpenAI(model="gpt-4o-mini")

    while True:
        query = input("Enter your query: ")
        if not query == "quit":
            answer = retriever.invoke(query)

            combined_input = f""" Based on the following documents, please answer this question {query}

            Documents:
            {'\n'.join([f"- {doc.page_content}" for doc in answer])}

            Provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents say "I don't have enough information to answer that question based on provided documents!"

            """

            messages = [
                SystemMessage(content="Your are a helpful assistant that answers questions based on the questions and conversation history provided."),
                
                HumanMessage(content=combined_input)
            ]

            response = model.invoke(messages)
            print(f"RAG Agent Response: \n {response.content}")

            
        else:
            break

if __name__ == "__main__":
    rag_resp()
