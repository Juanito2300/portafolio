import chromadb
from sentence_transformers import SentenceTransformer

# 🔥 Modelo de embeddings (rápido y muy bueno)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 Cliente persistente (MUY IMPORTANTE)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="portfolio",
    metadata={"hnsw:space": "cosine"}
)


def load_portfolio():
    """
    Carga el portfolio SOLO si la colección está vacía.
    Evita duplicados y hace tu API mucho más rápida.
    """

    if collection.count() > 0:
        print("✅ Portfolio ya cargado.")
        return

    print("⚡ Cargando portfolio en Chroma...")

    with open("portfolio.txt", encoding="utf-8") as f:
        text = f.read()

    # Mejor chunking (evita fragmentos basura)
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]

    ids = [str(i) for i in range(len(chunks))]
    embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print("🚀 Portfolio indexado correctamente.")


def search_context(query):
    """
    Busca los 3 fragmentos más relevantes.
    """

    if collection.count() == 0:
        load_portfolio()

    result = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=3
    )

    documents = result.get("documents", [[]])[0]

    if not documents:
        return "No se encontró contexto relevante."

    return "\n\n".join(documents)