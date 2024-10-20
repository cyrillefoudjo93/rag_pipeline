from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

# Singleton pattern for T5 model and tokenizer
_t5_model = None
_t5_tokenizer = None
_sentence_transformer = None

def get_t5_model():
    global _t5_model
    if _t5_model is None:
        _t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    return _t5_model

def get_t5_tokenizer():
    global _t5_tokenizer
    if _t5_tokenizer is None:
        _t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
    return _t5_tokenizer

# Singleton pattern for Sentence Transformer model
def get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _sentence_transformer
