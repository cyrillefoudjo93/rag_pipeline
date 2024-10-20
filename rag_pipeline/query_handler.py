from rag_pipeline.models import get_t5_model, get_t5_tokenizer
# import torch



def get_relevant_passages(query, index, texts, sentence_transformer, k=5):
    query_embedding = sentence_transformer.encode([query])
    _, indices = index.search(query_embedding, k)
    results = [texts[idx] for idx in indices[0]]
    return results


def generate_response(passages):
    t5_model = get_t5_model()
    t5_tokenizer = get_t5_tokenizer()
    combined_text = " ".join(passages)
    input_text = "summarize: " + combined_text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # with torch.no_grad():                 # Disable gradient calculation to speed up the process
    summary_ids = t5_model.generate(
        input_ids, 
        max_length=150,               # Increase this to make the response longer
        min_length=50,                # Set a minimum length to avoid very short summaries
        length_penalty=2.0,           # Lower values like 0.5 can produce longer texts
        num_beams=4, 
        early_stopping=True
    )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



# def generate_response(passages, chunk_size=300):
#     combined_text = " ".join(passages)
#     input_chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]
    
#     detailed_response = []
#     with torch.no_grad():
#         for chunk in input_chunks:
#             input_text = "summarize: " + chunk
#             input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
#             summary_ids = t5_model.generate(
#                 input_ids, 
#                 max_length=200,
#                 min_length=50, 
#                 length_penalty=1.0, 
#                 num_beams=4, 
#                 early_stopping=True,
#                 no_repeat_ngram_size=3
#             )
#             summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#             detailed_response.append(summary)
    
#     return " ".join(detailed_response)

