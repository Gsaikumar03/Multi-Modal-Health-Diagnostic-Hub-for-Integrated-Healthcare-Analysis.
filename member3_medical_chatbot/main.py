import os
from src.pdf_loader import load_pdf_text
from src.text_cleaner import clean_text, chunk_text
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.rag_pipeline import MedicalRAG

PDF_PATH = "data/medical.pdf"
INDEX_PATH = "faiss_index.bin"
META_PATH = "texts.pkl"

def build_vector_database():

    print("\nBuilding FAISS Vector Database (Week 1 + 2)...")

    # Load PDF
    raw_text = load_pdf_text(PDF_PATH)
    print("Raw text length:", len(raw_text))

    # Clean Text
    cleaned_text = clean_text(raw_text)
    print("Cleaned text length:", len(cleaned_text))

    # Chunk Text
    chunks = chunk_text(cleaned_text)
    print("Number of text chunks:", len(chunks))

    print("\nSample Chunk:\n")
    print(chunks[0])

    # Embeddings + FAISS
    embedder = EmbeddingModel()
    embeddings = embedder.encode(chunks)

    dimension = embeddings.shape[1]

    vector_store = FAISSVectorStore(dimension)
    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save(INDEX_PATH, META_PATH)

    print("\nFAISS index built and saved.\n")



# MAIN FUNCTION 
def main():

    # If FAISS index not present → build it
    if not os.path.exists(INDEX_PATH):
        build_vector_database()
    else:
        print("FAISS index found. Loading existing database...\n")

    # Initialize Hybrid RAG 
    rag = MedicalRAG()

    print("Hybrid Medical RAG System Ready ✅")
    print("Using FAISS + Keyword Re-ranking (Week 5)")
    print("Confidence scoring enabled (Week 6)\n")

    # Chat Loop
    while True:
        question = input("Ask a medical question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Exiting chatbot...")
            break

        response = rag.ask(question)

        print("\n==============================")
        print("Answer:\n")
        print(response["answer"])

        print("\nConfidence Score:", response["confidence"])

        print("\n--- Retrieved Context Chunks Used ---")
        for i, chunk in enumerate(response["context_used"]):
            print(f"\nChunk {i+1} Preview:\n{chunk[:200]}...")

        print("==============================\n")



# ENTRY POINT
if __name__ == "__main__":
    main()