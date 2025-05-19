import os
import warnings
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Could not reliably determine page label for")

# 🔹 Carica le chiavi API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quickstart")

# 🔹 Inizializza Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# 🔹 Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 🔹 Funzione per dividere una lista in batch di N elementi
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

# 🔹 Funzione per caricare un singolo PDF
def load_and_upload_pdf(filepath):
    print(f"📄 Caricamento file: {filepath}")
    loader = PyPDFLoader(filepath)
    pages = loader.load()
	
	# 🔸 Pulizia del testo per rimuovere parole spezzate
    for page in pages:
        page.page_content = page.page_content.replace("-\n", "")  # unisce parole spezzate
        page.page_content = page.page_content.replace("\n", " ")  # uniforma il testo

    
	
	
    # 🔸 Suddivide il testo
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    #chunks = text_splitter.split_documents(pages)

    # 🔸 Unisce tutto il testo in una stringa
    full_text = " ".join([page.page_content for page in pages])

    # 🔸 Suddivide usando numerazioni o titoli in maiuscolo (es. "3 - 6 IT" oppure "CONTACHILOMETRO PARZIALE")
    import re
    raw_chunks = re.split(
        r"(?:\n|\r)?\s*\d+\s*-\s*\d+\s*[A-Z]{2}\s*|\n{2,}|(?<=\n)[A-Z][^\n]{5,100}(?=\n)",
        full_text
    )

    # 🔸 Pulisce e filtra i blocchi
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk and len(chunk.strip()) > 100]

    # 🔸 Suddivisione ulteriore se necessario (evita chunk troppo grandi)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(cleaned_chunks)
	
	
	
	
	# 🔸 Nome del file → namespace
    filename = os.path.basename(filepath)
    namespace = os.path.splitext(filename)[0]

    print(f"🔹 Namespace: {namespace}, Chunk totali: {len(chunks)}")

    # 🔸 Caricamento a batch
    batch_size = 100  # puoi abbassarlo a 30 se necessario
    for batch in tqdm(list(batch_iterable(chunks, batch_size)), desc="Caricamento batch"):
        PineconeVectorStore.from_documents(
            batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=namespace
        )

    print(f"✅ Caricato '{filename}' con successo.\n")

# 🔹 Carica tutti i PDF nella cartella ./pdfs
pdf_folder = "./pdfs"
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        load_and_upload_pdf(os.path.join(pdf_folder, file))
        time.sleep(10)  # pausa dopo ogni file