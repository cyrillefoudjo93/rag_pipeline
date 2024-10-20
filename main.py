from rag_pipeline.extraction import extract_texts_from_pdfs
from rag_pipeline.vector_store import create_embeddings, store_embeddings_in_faiss
from rag_pipeline.query_handler import get_relevant_passages, generate_response
from rag_pipeline.models import get_sentence_transformer


def get_answer(query: str):
    try:
        # Load a pre-trained model for creating embeddings (e.g., multilingual model)
        print("Load pre-trained Sentence Transformer Model...")
        sentence_transformer = get_sentence_transformer()

        directory = './pdfs'
        print("Extrahiere Texte aus PDFs...")
        text_sections = extract_texts_from_pdfs(directory)
        print("Texte extrahiert")
        
        print("Erstelle Embeddings...")
        embeddings = create_embeddings(text_sections, sentence_transformer)
        print("Embeddings erstellt")
        
        print("Speichere Embeddings in FAISS...")
        index, texts = store_embeddings_in_faiss(embeddings, text_sections)
        print("Embeddings gespeichert")
        
        print("Führe Abfrage aus:", query)
        passages = get_relevant_passages(query, index, texts, sentence_transformer)
        print("Relevante Passagen gefunden")
        
        response = generate_response(passages)
        
        print("Antwort auf die Anfrage:")
        print(response)
    except Exception as e:
        print("Ein Fehler ist aufgetreten:", e)

# def main():
#     directory = './pdfs'
#     texts = extract_texts_from_pdfs(directory)
#     embeddings = create_embeddings(texts)
#     index, text_sections = store_embeddings_in_faiss(embeddings, texts)
    
#     query = "Wie hoch ist die Grundzulage?"
#     # query = "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?"
#     # query = "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
#     passages = get_relevant_passages(query, index, texts)
#     response = generate_response(passages)
    
#     print("Frage:", query, ":\n\n", response)
#     print(response)

if __name__ == "__main__":
    # query = "Wie hoch ist die Grundzulage?"
    # query = "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?"
    # query = "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    get_answer(query="Wie hoch ist die Grundzulage?")
