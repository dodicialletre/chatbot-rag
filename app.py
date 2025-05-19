import os
from flask import Flask, request, render_template
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from flask import jsonify

# 🔹 Carica le variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quickstart")

# 🔹 Inizializza Pinecone e modelli
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#openai.api_key = OPENAI_API_KEY

# 🔹 Inizializza client OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# 🔹 Flask App
app = Flask(__name__)

# Funzione per ottenere tutti i namespace presenti nell'indice
def get_namespaces():
    stats = pinecone_index.describe_index_stats()
    return list(stats.get("namespaces", {}).keys())
	
# 🔹 Funzione di ricerca Pinecone con filtro score
def search_documents(query, namespace, score_threshold=0.75):
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )
    results = vectorstore.similarity_search_with_score(query, k=5)

    # Filtra i risultati per punteggio
    filtered_docs = [doc for doc, score in results if score >= score_threshold]

    print(f"\n🔍 Documenti sopra la soglia ({score_threshold}): {len(filtered_docs)}")
    for doc, score in results:
        print(f"Score: {score:.2f} - Anteprima: {doc.page_content[:200]}")

    return filtered_docs

# 🔹 Funzione per generare risposta con contesto
def generate_answer_with_context(query, docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Sei un assistente tecnico esperto. Usa le informazioni qui sotto per rispondere in modo chiaro e preciso alla domanda.\n\n"
        f"Informazioni:\n{context}\n\n"
        f"Domanda: {query}\n"
        f"Risposta:"
    )

    response = client.chat.completions.create(
        model="gpt-4o",  # oppure gpt-4 o gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

@app.route("/namespaces", methods=["GET"])
def list_namespaces():
    namespaces = get_namespaces()
    return jsonify({"namespaces": namespaces})
	
# 🔹 Route homepage
@app.route("/", methods=["GET"])
def index():
    namespaces = get_namespaces()
    return render_template("index.html", namespaces=namespaces)

# 🔹 Route per ricevere la domanda
@app.route("/ask", methods=["POST"])
def ask():
    query = request.form["query"]
    model = request.form["model"]
    namespaces = get_namespaces()

    results = search_documents(query, model, score_threshold=0.75)

    if results:
        answer = generate_answer_with_context(query, results)
    else:
        answer = "Nessuna risposta trovata."
		
    return render_template("index.html", question=query, answer=answer, namespaces=namespaces, selected=model)

if __name__ == "__main__":
    app.run(debug=True)