from dotenv import load_dotenv
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import chromadb

class PDFReaderDB:
  # TODO: eu acho que a gente deveria criar uma classe
  def __init__(self, embeddings):
    self.embeddings = embeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

def add_collection(file_path, collection_name):
  load_dotenv('.env.local')

  storage_path = os.getenv('STORAGE_PATH')
  if storage_path is None:
      raise ValueError('STORAGE_PATH environment variable is not set')

  if not os.path.isdir(storage_path):
    raise NotADirectoryError('STORAGE_PATH must be a repository')

  # Local PDF file uploads
  print("Reading pdf...")
  loader = UnstructuredPDFLoader(file_path=file_path)
  data = loader.load()
  print("Done!")

  # Split and chunk 
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  chunks = text_splitter.split_documents(data)

  # Add to vector database
  vector_db = Chroma.from_documents(
      documents=chunks, 
      embedding=embeddings,
      collection_name=collection_name,
      persist_directory=storage_path
  )

  print(f'File {file_path} uploaded to collection {collection_name}')

def get_vector_store(collection_name):
  persistent_client = chromadb.PersistentClient(path='chromadb')
  return Chroma(client=persistent_client,
                collection_name=collection_name,
                embedding_function=embeddings)

add_collection('pdf_files/owner_manual_p283-p300.pdf', 'short_manual')
