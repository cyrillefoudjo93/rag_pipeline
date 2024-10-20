import faiss
import numpy as np


def create_embeddings(text_sections, sentence_transformer):
    embeddings = sentence_transformer.encode(text_sections)
    return np.array(embeddings)

def store_embeddings_in_faiss(embeddings, text_sections):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, text_sections
