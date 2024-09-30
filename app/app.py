import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# load vector store
def get_vector_store(collection_name):
  persistent_client = chromadb.PersistentClient(path='chromadb')
  return Chroma(client=persistent_client,
                embedding_function=embeddings,
                collection_name=collection_name)

# Title of the app
st.title("Local LLM Text Generator")

# Sidebar information
#st.sidebar.header("Model Settings")

# Load the local LLM model (GPT-2 or any other HuggingFace model)
@st.cache_resource
def load_model():   
    # embeddings
    

    vector_db = get_vector_store('owner_manual')
    retriever = vector_db.as_retriever()

    # LLM from Ollama
    local_model = "mistral"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You will answer questions about information that can be found in the owner's manual of the RAM 1500 vehicle, model year 2025, Crew Cab version.
    Your task is to generate five different versions of the given user question to retrieve relevant documents
    from a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return chain

# Load model
chain = load_model()

# Model settings (temperature, max tokens, etc.)
#temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
#max_length = st.sidebar.slider("Max Length", 50, 300, 100)

# Text input for the user prompt
prompt = st.text_area("Enter your prompt:", "Ask a question:")

# Generate button
if st.button("Generate"):
    # Generate text
    generated_text = chain.invoke(prompt)

    # Display the generated text
    st.subheader("Generated Text:")
    st.write(generated_text)
