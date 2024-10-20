import faiss
import numpy as np
from rag_pipeline.models import get_sentence_transformer

model = get_sentence_transformer()

def create_embeddings(text_sections):
    embeddings = model.encode(text_sections)
    return np.array(embeddings)

def store_embeddings_in_faiss(embeddings, text_sections):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_sections
