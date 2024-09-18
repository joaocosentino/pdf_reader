from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv('.env.local')

storage_path = os.getenv('STORAGE_PATH')
if storage_path is None:
    raise ValueError('STORAGE_PATH environment variable is not set')

if not os.path.isdir(storage_path):
   raise NotADirectoryError('STORAGE_PATH must be a repository')

local_path = "pdf_files/Owners_Manual-Ram_1500_25_Crew_Cab.pdf"

# Local PDF file uploads
if local_path:
  print("Reading pdf...")
  loader = UnstructuredPDFLoader(file_path=local_path)
  data = loader.load()
else:
  print("Upload a PDF file")

# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="owner_manual",
    persist_directory=storage_path
)