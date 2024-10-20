from rag_pipeline.extraction import extract_texts_from_pdfs
from rag_pipeline.vector_store import create_embeddings, store_embeddings_in_faiss
from rag_pipeline.query_handler import get_relevant_passages, generate_response

def main():
    directory = './pdfs'
    texts = extract_texts_from_pdfs(directory)
    embeddings = create_embeddings(texts)
    index, text_sections = store_embeddings_in_faiss(embeddings, texts)
    
    query = "Wie hoch ist die Grundzulage?"
    # query = "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?"
    # query = "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    passages = get_relevant_passages(query, index, texts)
    response = generate_response(passages)
    
    print("Frage:", query, ":\n\n", response)
    print(response)

if __name__ == "__main__":
    main()
