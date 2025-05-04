from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# step 1: Upload and load raw PDF(s)
pdfs_directory = 'pdfs/'

# will store file as it is inside the pdfs directory
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents



# step 2: Create Chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

file_path='universal_rights.pdf'
documents=load_pdf(file_path=file_path)
# print("pdf pages: ", len(documents))
text_chunks = create_chunks(documents=documents)
# print('chunks count: ', len(text_chunks))

# Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
ollama_model_name='deepseek-r1:7b'
def get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings



# step 4: Index Documents "Store embeddings in FAISS (vector Database)"
FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)
