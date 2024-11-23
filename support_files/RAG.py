import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import pdfplumber
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


def extract_pdf_by_sections(pdf_path):
    """
    This function create a list of Document objects that will be passed to a vector database as embeddings.
    """
    current_section = {"title": None, "content": ""}
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Get character-level data
            creating_title = False
            prev_y = None
            for char in page.chars:
                text = char.get("text", "")
                font_name = char.get("fontname", "")
                is_bold = "Bold" in font_name  # Adjust based on your PDF's font styles
                current_y = char["top"]

                if is_bold and not(creating_title):
                    # If we encounter bold text and already have a section, close it
                    if current_section["title"]:
                        document = Document(page_content=current_section["content"],
                                            metadata={"page": page_number,
                                                      "section": current_section["title"]})
                        documents.append(document)
                        current_section = {"title": None, "content": ""}
                        prev_y = None

                    # Start a new section
                    current_section["title"] = text
                    creating_title = True
                elif is_bold and creating_title:
                    current_section["title"] += text
                else:
                    # Add content to the current section
                    creating_title = False

                    if prev_y is not None:
                        diff_in_height = abs(prev_y-current_y)
                        if diff_in_height > 9:
                            current_section["content"] += "\n"
                        elif diff_in_height > 0.5:
                            current_section["content"] += " "
                    prev_y = current_y

                    current_section["content"] += text

    # Append the last section, if any
    if current_section["title"]:
        document = Document(page_content=current_section["content"],
                            metadata={"page": page_number,
                                      "section": current_section["title"]})
        documents.append(document)

    return documents

class RAGSystem:
    def __init__(self, persist_directory: str = "db", embedding_model="nomic-embed-text", llm_model="mistral"):
        """
        Initialize the RAG System with embeddings, retriever, and LLM.
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.embedding_function = OllamaEmbeddings(model=self.embedding_model, show_progress=True)
        self.vector_store = None
        self.retriever = None
        self.llm = ChatOllama(model=self.llm_model)
        self.qa_chain = None
    
    def process_documents(self, file_paths: list, collection_name="langchain"):
        """
        Process documents and load them into a vector database.
        """
        documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                document = extract_pdf_by_sections(file_path)
                documents.extend(document)
            else:
                print(f"Unsupported file type: {file_path}")
        
        self.vector_store = Chroma.from_documents(documents,
                                                  self.embedding_function,
                                                  persist_directory=self.persist_directory,
                                                  collection_name=collection_name)
        self.vector_store.persist()
        self.retriever = self.vector_store.as_retriever()
        print(f"Processed and saved {len(documents)} documents.")
    
    def load_existing_vector_store(self, collection_name="langchain"):
        """
        Load an existing vector store from the persist directory.
        """
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function, collection_name=collection_name)
            self.retriever = self.vector_store.as_retriever()
            print("Loaded existing vector store.")
        else:
            raise FileNotFoundError("Persist directory not found. Please process documents first.")
    
    def generate_query(self, user_input: str) -> str:
        """
        Optionally preprocess or enhance the user query.
        """
        # Add custom query enhancement logic here if needed
        return user_input
    
    def answer_query(self, query: str) -> str:
        """
        Generate an answer for the given query.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Load or process documents first.")
        
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Your task is to generate five different versions of the given user question to retrieve relevant documents
            from a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        
        retriever = MultiQueryRetriever.from_llm(
            self.retriever, 
            self.llm,
            prompt=QUERY_PROMPT
        )

        # RAG prompt
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(query)

        return answer
    
    def evaluate_queries(self, query_file: str, output_file: str):
        """
        Evaluate a list of queries from an Excel file and save answers to another file.
        """
        queries_df = pd.read_excel(query_file)
        if "query" not in queries_df.columns:
            raise ValueError("The input file must contain a 'query' column.")
        
        queries_df["answer"] = queries_df["query"].apply(self.answer_query)
        queries_df.to_excel(output_file, index=False)
        print(f"Answers saved to {output_file}")

# test code
rag = RAGSystem()
rag.process_documents(["./pdf_files/owner_manual_p283-p300.pdf"])

query = "Where is the hazard warning flashers button?"
answer = rag.answer_query(query)
print("Answer:", answer)

input_queries = "./support_files/testing_queries.xlsx"
output_answers = "./support_files/answers.xlsx"
rag.evaluate_queries(input_queries, output_answers)

