import os
import sys
import uuid
import socket
import chromadb
import traceback
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import (
        TextLoader, 
        PyPDFLoader, 
        WebBaseLoader,
        Docx2txtLoader, 
        UnstructuredExcelLoader, 
        UnstructuredPowerPointLoader
    )
from config import environment as env, llms
from langchain_ollama import OllamaEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

cols4 = st.columns([0.8, 10, 1.5])

def clear_messages():
    st.session_state.document_messages = st.session_state.document_messages[:1]

with cols4[0]:
    st.image(image = str(Path(env.ASSETS_DIR, "chat_with_document.png")), width = 75, use_container_width = True)
with cols4[1]:
    st.header("Chat with Document", divider = "red", anchor = False)
with cols4[2]:
    st.button("Clear Chat", on_click = lambda: clear_messages(), type = "primary")


# Initialize chat session in streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("LLM Model")

    col1, col2 = st.columns([0.5, 0.5])
    model_type = st.radio("Models", options = ["Ollama", "Google Gemini"], key = "model_type", horizontal = True, index = 1, label_visibility = "collapsed")
    
llm_models: dict[str] = {}

if model_type == "Ollama":
    API_URL = "http://127.0.0.1:11434/"
    try:
        ollama_models = llms.get_ollama_models()
    except Exception as e:
        st.error(f"ERROR: {e}", icon = "⚠️")
        st.stop()

    if len(ollama_models) == 0:
        st.info(f"""No models found...! Please download a model from Ollama library to proceed.

Command:

ollama pull <model name>
                
You can visit the website: https://ollama.com/library to get models names.
""", icon = "⚠")
        st.stop()
          
    else:
        for i, model in enumerate(list(ollama_models.values())):
            if "llama3.1" in model:
                default_model_index: int = i
                break
            else:
                default_model_index: int = None

        ollama_model_name = list(ollama_models.keys())
        for model in ollama_models:
            llm_models[model] = ollama_models[model]

elif model_type == "Google Gemini":
    google_models = llms.get_google_models(modality = "text")
    
    # Get default index of models
    for i, model in enumerate(google_models.values()):
        if "gemini-1.5-flash" in model:
            default_model_index: int = i
            break
        else:
            default_model_index: int = None

    google_model_name = list(google_models.keys())
    for model in google_models:
        llm_models[model] = google_models[model]

else:
    st.error("Invalid model type selected. Please choose either Ollama or Google Gemini.", icon = "🚫")
    st.stop()

with st.sidebar:
    selected_model = llm_models[st.selectbox("Model Name", options = ollama_model_name if model_type == "Ollama" else google_model_name, label_visibility = "collapsed", key = "selected_model", index = default_model_index if default_model_index is not None else 0)]
    streaming = st.toggle(label="Streaming output", key = "streaming", value = True, help = "Enable streaming output for the assistant's response.")
    history_flag = st.toggle(label="Chat History", key="history", value = True, help = "Enable chat history or memory for the assistant's response.")
    st.session_state.embed_provider = st.selectbox(label = "Embedding Provider", options = ["Ollama", "Google"])

    st.divider()
    st.markdown("## Session ID")
    st.markdown(f":grey[{st.session_state.session_id}]")
    st.markdown("## Instructions")
    st.markdown(":grey[You can chat with a document to get answers to questions, summarize content, or extract information using RAG.]\n\n")
    st.markdown("## About")
    st.markdown(":grey[This app is built using Streamlit, Ollama and Google Gemini.]")
    st.divider()

    st.write(f""":grey[Hostname: {socket.gethostname()}]<br>
                :grey[IP: {socket.gethostbyname(socket.gethostname())}]""", unsafe_allow_html = True)

def get_llm():
    if model_type == "Ollama":
        llm = ChatOllama(model = selected_model, base_url = API_URL)
    elif model_type == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model = selected_model)
    return llm

def document_loader(document_files: list):
    """Load a document file and return its content."""
    documents: list = []

    # Create or set a temporary directory to store the uploaded files
    os.makedirs(env.TEMP_DIR, exist_ok = True)

    for file in document_files:
        temp_file_path = Path(env.TEMP_DIR, file.name)
        file_extension: str = file.name.split(".")[-1].lower()
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.read())

        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension in ["txt", "md", "py"]:
                loader = TextLoader(temp_file_path, encoding = "utf-8")
            elif file_extension == "xlsx":
                loader = UnstructuredExcelLoader(temp_file_path)
            elif file_extension == "pptx":
                loader = UnstructuredPowerPointLoader(temp_file_path)
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}", icon = "⚠️")
            st.stop()
        finally:
            documents.extend(loader.load())
            os.remove(temp_file_path)
    return documents

def url_loader(url):
    documents: list = []
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    documents.extend(loader.load())
    return documents
    
def split_documents(documents) -> list[str]:
    """ Load a text file and split it into chunks."""
    chunk_size = 2000
    chunk_overlap = 300
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    """Get the embeddings model based on the selected model type."""
    if st.session_state.embed_provider == "Ollama":
        embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    elif st.session_state.embed_provider == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-exp-03-07")
    return embeddings

def create_vectordb(documents: list, collection_name) -> Chroma:
    """Create a vector database from the documents using Google Generative AI embeddings."""
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(documents, embedding = embeddings, collection_name = collection_name, persist_directory = str(env.CHROMADB_DIR))
    return vectordb

def get_existing_vectordb(collection_name) -> Chroma:
    embeddings = get_embeddings()
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(env.CHROMADB_DIR)))
    vectordb = Chroma(client = client, collection_name = collection_name, embedding_function = embeddings)
    return vectordb

def list_collections():
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(env.CHROMADB_DIR)))
    collections = [collection.name for collection in client.list_collections()]
    return collections

def delete_collection(collection_name):
    client = chromadb.Client(Settings(is_persistent = True, persist_directory = str(env.CHROMADB_DIR)))
    client.delete_collection(name = collection_name)

def get_embedded_documents(vector_db: Chroma) -> list:
    metadata: list[dict] = vector_db.get()["metadatas"]
    file_list = []
    for file in metadata:
        if str(file['source']).startswith("http"):
            file_list.append(file["source"])
        else:
            file_list.append(Path(file["source"]).name)
    embedded_files = list(set(file_list))
    return embedded_files

def get_history_aware_retriever(vector_db: Chroma, llm):
    """Create a history-aware retriever chain using the vector database and LLM."""
    # The prompt used to generate the search query for the retriever.
    retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
            ])
    retriever = vector_db.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, retriever_prompt)
    return retriever_chain

def get_retrieval_chain(vector_db: Chroma, llm):
    # Prompt template MUST contain input variable “context” (override by setting document_variable), which will be used for passing in the formatted documents.
    document_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. You will have to answer to user's prompts.
                        You will have some context to help with your answers, but it might not always would be completely related or helpful. Do not quote references from the document provided, respond in your own language.
                        You can also use your knowledge to assist answering the user's propmts.\n
            {context}"""),
            MessagesPlaceholder(variable_name = "chat_history"),
            ("user", "{input}"),
        ])
    retriever = get_history_aware_retriever(vector_db, llm)
    stuff_documents_chain = create_stuff_documents_chain(llm, document_prompt)
    retriever_chain = create_retrieval_chain(retriever, stuff_documents_chain)
    return retriever_chain

if "embedded_documents" not in st.session_state:
    st.session_state.embedded_documents = []
    

def load_rag_sources(collection_name, source_type):
    with st.spinner("Embedding Document..."):
        
        if source_type == "Documents":
            if st.session_state.selected_files:                
                documents = document_loader(st.session_state.selected_files)
        elif source_type == "URL":
            if st.session_state.selected_url:
                documents = url_loader(st.session_state.selected_url)
        
        # Create chunks from the documents
        chunks = split_documents(documents)
        if len(chunks) > 0:
            try:
                create_vectordb(chunks, collection_name)
                if source_type == "Documents":
                    st.toast(f"Document *{str([file.name for file in st.session_state.selected_files])[1:-1]}* loaded successfully", icon = "✅")
                elif source_type == "URL":
                    st.toast(f"URL *{str(st.session_state.selected_url)}* loaded successfully", icon = "✅")

            except Exception as e:
                traceback_str = traceback.format_exception(e)
                st.error(traceback_str)
                st.stop()
        else:
            st.toast(f"ⓘ Document *{str([file.name for file in st.session_state.selected_files])[1:-1] if source_type == 'Documents' else st.session_state.selected_url}* already loaded in Collection")
            del st.session_state.show_add_document

if "show_add_document" not in st.session_state:
    st.session_state.show_add_document = False

def show_add_document():
    st.session_state.show_add_document = True

def hide_add_document():
    st.session_state.show_add_document = False

llm = get_llm()

col1, col2 = st.columns([0.35, 0.65])

col1.markdown("#### :blue[Document Collections]")

collection_list = list_collections()

cols1 = col1.columns([0.6, 0.1, 0.1])
collection = cols1[0].selectbox("Collection", options = collection_list, key = "collection", label_visibility = "collapsed", index = 0, placeholder = "Select a Collection")
cols1[1].button(":material/add:", on_click = show_add_document, help = "Add Document to Selected Collection", use_container_width = True)
cols1[2].button(":material/delete:", type = "primary", on_click = lambda: delete_collection(collection_name = st.session_state.collection), help = "Delete Selected Collection", disabled = True if collection is None else False, use_container_width = True)

is_vector_db_loaded = False

if st.session_state.collection is not None:
    st.session_state.vector_db = get_existing_vectordb(st.session_state.collection)
    chain = get_retrieval_chain(st.session_state.vector_db, llm)
    is_vector_db_loaded = True if len(st.session_state.collection) > 0 else False

elif len(collection_list) == 0:
    col1.info("⚠︎ No documents loaded. Please add new document to chat with.")

col1.toggle(
    "Use RAG",
    value = is_vector_db_loaded,
    key = "use_rag",
    disabled = not is_vector_db_loaded
)

config_container = col1.container(height = 300 if len(collection_list) > 0 else 200, border = False)

if st.session_state.collection is not None:
    st.session_state.embedded_docs = get_embedded_documents(get_existing_vectordb(collection_name = st.session_state.collection))
    with config_container.expander(label = f":grey[Documents in Collection ({len(st.session_state.embedded_docs)})]", expanded = False):
        st.write(st.session_state.embedded_docs)

if st.session_state.show_add_document:
    col3, col4, col5 = config_container.columns([0.7, 0.1, 0.01])
    col3.markdown("#### :blue[Add or Create RAG Sources]")
    col4.button(":material/close:", type = "tertiary", on_click = hide_add_document, help = "Close", use_container_width = True)
    new_collection_name = config_container.text_input("New Collection Name", value = None, placeholder = "New Collection Name", label_visibility = "collapsed")
    source_type = config_container.radio("Source Type", options = ["Documents", "URL"], horizontal = True, label_visibility = "collapsed")

    if source_type == "Documents":
        uploaded_flg = False
        config_container.file_uploader(
            "🗎 Upload a document",
            type = ["pdf", "txt", "docx", "md", "xlsx", "pptx"],
            accept_multiple_files = True,
            key = "selected_files",
            help = "Upload a document to chat with it.",
            label_visibility = "collapsed"
        )
        if len(collection_list) == 0:
            if len(st.session_state.selected_files) > 0 and new_collection_name is not None and new_collection_name != "":
                uploaded_flg = True
        else:
            if len(st.session_state.selected_files) > 0:
                uploaded_flg = True
        
        config_container.button(f"Add Files to collection: '{st.session_state.collection}'" if new_collection_name is None or new_collection_name == "" else f"Create new collection: '{new_collection_name.replace(' ', '_')}'", on_click = lambda: load_rag_sources(collection_name = st.session_state.collection if new_collection_name is None or new_collection_name == "" else new_collection_name.replace(' ', '_'), source_type = "Documents"), disabled = not uploaded_flg)
        
        
    if source_type == "URL":
        uploaded_flg = False
        config_container.text_input("🌐︎ Provide a URL", key = "selected_url", placeholder = "https://example.com", label_visibility = "collapsed")

        if len(collection_list) == 0:
            if len(st.session_state.selected_url) > 0 and new_collection_name is not None and new_collection_name != "":
                uploaded_flg = True
        else:
            if len(st.session_state.selected_url) > 0:
                uploaded_flg = True
                
        config_container.button(f"Add URL to collection: '{st.session_state.collection}'" if new_collection_name is None or new_collection_name == "" else f"Create new collection: '{new_collection_name.replace(' ', '_')}'", on_click = lambda: load_rag_sources(collection_name = st.session_state.collection if new_collection_name is None or new_collection_name == "" else new_collection_name.replace(' ', '_'), source_type = "URL"), disabled = not uploaded_flg)


# Get Conversational RAG chain
if "document_messages" not in st.session_state:
    st.session_state.document_messages = [{
        "role": "assistant",
        "content": "Hi there! How can I help you today?"
    }]

document_message_container = col2.container(height = 450, border = True)

for message in st.session_state.document_messages:
    with document_message_container:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_prompt := st.chat_input(f"Ask {st.session_state.selected_model.split(' ')[0].title()}"):
    user_input: dict[str, list] = {"role": "user", "content": user_prompt}
    st.session_state.document_messages.append(user_input)

    with document_message_container:
        with st.chat_message(user_input["role"]):
            st.markdown(user_input["content"])

        with st.chat_message("assistant"):
            with st.spinner(":grey[Generating...]"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    if st.session_state.use_rag:
                        if streaming:
                            for chunk in chain.pick("answer").stream({"chat_history": st.session_state.document_messages[:-1] if history_flag else [{"role": "user", "content": ""}], "input": st.session_state.document_messages[-1]["content"]}):
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                        else:
                            response = chain.invoke({"chat_history": st.session_state.document_messages[:-1] if history_flag else [{"role": "user", "content": ""}], "input": st.session_state.document_messages[-1]["content"]})
                            full_response = response["answer"]
                    else:
                        if streaming:
                            for chunk in llm.stream(st.session_state.document_messages):
                                full_response += chunk.content
                                message_placeholder.markdown(full_response + "▌")
                        else:
                            response = llm.invoke(st.session_state.document_messages)
                            full_response = response.content
                        
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    full_response = f":red[Error: {traceback_str}]"
                    # full_response = f":red[Error: {e}]"

            message_placeholder.markdown(full_response)
            st.session_state.document_messages.append({"role": "assistant", "content": full_response})

