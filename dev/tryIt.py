# import os
# import numpy as np
# import faiss
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.text_rank import TextRankSummarizer
# import fitz  # PyMuPDF for extracting text from PDFs
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Ensure NLTK and Dependencies are Installed: Make sure you have NLTK installed correctly in the actual virtual environment

# import nltk
# import ssl
# import certifi

# # Set the SSL context
# ssl._create_default_https_context = ssl._create_unverified_context

# # Now download the punkt data
# try:
#     nltk.download('punkt')
#     nltk.download('punkt_tab')  # specifically download punkt_tab
# except Exception as e:
#     print(f"Error downloading NLTK data: {e}")


# def extract_text_from_pdfs(pdf_folder):
#     """
#     Extracts text from all PDF files in a folder.
#     """
#     all_texts = []
#     for filename in os.listdir(pdf_folder):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(pdf_folder, filename)
#             with fitz.open(file_path) as pdf_file:
#                 text = ""
#                 for page in pdf_file:
#                     text += page.get_text()
#                 all_texts.append(text)
#     return all_texts

# def break_into_paragraphs(text, min_length=50):
#     """
#     Breaks the extracted text into paragraphs or sections.
#     """
#     paragraphs = text.split("\n")
#     meaningful_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_length]
#     return meaningful_paragraphs

# def create_vector_database(paragraphs):
#     """
#     Creates a vector database for the given paragraphs using TF-IDF vectors.
#     """
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(paragraphs).toarray()

#     # Create FAISS index
#     dimension = vectors.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # Using L2 distance
#     index.add(np.array(vectors).astype('float32'))  # Convert to float32 for FAISS

#     return index, vectorizer

# def summarize_text(text, sentence_count=3):
#     """
#     Summarizes the given text using the TextRank algorithm from the `sumy` library.
#     """
#     parser = PlaintextParser.from_string(text, Tokenizer("english"))
#     summarizer = TextRankSummarizer()
#     summary = summarizer(parser.document, sentence_count)
#     summarized_response = " ".join(str(sentence) for sentence in summary)
#     return summarized_response

# def query_vector_db(query, index, vectorizer):
#     """
#     Queries the vector database and retrieves the most relevant paragraphs based on the query.
#     """
#     query_vector = vectorizer.transform([query]).toarray().astype('float32')
#     _, indices = index.search(query_vector, k=5)  # Retrieve top 5 paragraphs
#     return indices.flatten()

# def main(pdf_folder, query):
#     # Step 1: Extract text from the PDF documents
#     print("Extracting text from PDF files...")
#     all_texts = extract_text_from_pdfs(pdf_folder)

#     # Combine all extracted texts into a single string
#     combined_text = " ".join(all_texts)

#     # Step 2: Break the extracted text into meaningful paragraphs
#     print("Breaking text into paragraphs...")
#     paragraphs = break_into_paragraphs(combined_text)

#     # Step 3: Create a vector database for the paragraphs
#     print("Creating vector database...")
#     index, vectorizer = create_vector_database(paragraphs)

#     # Step 4: Query the vector database
#     print(f"Querying for: {query}")
#     indices = query_vector_db(query, index, vectorizer)

#     # Retrieve and summarize the most relevant paragraphs
#     relevant_paragraphs = [paragraphs[i] for i in indices]
#     combined_relevant_text = " ".join(relevant_paragraphs)

#     # Step 5: Summarize the relevant paragraphs
#     print("Summarizing the relevant paragraphs...")
#     summarized_text = summarize_text(combined_relevant_text, sentence_count=3)
    
#     return summarized_text


# import os
# import fitz  # PyMuPDF
# import numpy as np
# import faiss  # For vector storage
# import spacy
# from transformers import pipeline

# # Load the German language model in spaCy
# nlp = spacy.load("de_core_news_sm")

# # Load a transformer model for summarization
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Custom mapping of complex adjectives to simpler alternatives
# simplification_dict = {
#     "schwierig": "einfach",
#     "kompliziert": "einfach",
#     "herausfordernd": "einfach",
#     # Add more mappings as needed
# }

# def extract_text_from_pdfs(pdf_folder):
#     texts = []
#     for filename in os.listdir(pdf_folder):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(pdf_folder, filename)
#             pdf_document = fitz.open(pdf_path)
#             text = ""
#             for page in pdf_document:
#                 text += page.get_text()
#             texts.append(text)
#     return texts

# def create_embeddings(texts):
#     # Dummy embedding function, replace with your embedding method
#     return np.random.rand(len(texts), 512).astype('float32')  # 512-dimensional vectors

# def store_in_vector_db(embeddings):
#     index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
#     index.add(embeddings)
#     return index

# def query_vector_db(query, index, texts):
#     query_embedding = create_embeddings([query])  # Replace with actual embedding logic
#     D, I = index.search(query_embedding, k=3)  # Retrieve top 3 closest vectors
#     return [texts[i] for i in I[0]]

# # def summarize_text(text, max_length=130):
# #     # Use the transformer model to summarize
# #     summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
# #     return summary[0]['summary_text']

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load T5 model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")

# import torch
# device = torch.device('cpu') 
# model.to(device)

# def summarize_text(text, max_length=130):
#     input_text = f"summarize: {text}"
#     input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

#     # Generate summary
#     summary_ids = model.generate(input_ids, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary


# def simplify_adjectives(text):
#     doc = nlp(text)
#     simplified_text = text

#     for token in doc:
#         if token.pos_ == "ADJ":  # Check if the token is an adjective
#             simpler_word = simplification_dict.get(token.text)
#             if simpler_word:
#                 # Replace the adjective with its simpler alternative
#                 simplified_text = simplified_text.replace(token.text, simpler_word, 1)

#     return simplified_text

# def main(pdf_folder, query):
#     # Step 1: Extract text from PDF documents
#     texts = extract_text_from_pdfs(pdf_folder)

#     # Step 2: Create embeddings for the texts
#     embeddings = create_embeddings(texts)

#     # Step 3: Store embeddings in a vector database
#     index = store_in_vector_db(embeddings)

#     # Step 4: Query the vector database
#     relevant_texts = query_vector_db(query, index, texts)

#     # Step 5: Summarize the relevant texts
#     combined_relevant_text = " ".join(relevant_texts)
#     summarized_text = summarize_text(combined_relevant_text)

#     # Step 6: Simplify the language
#     simplified_text = simplify_adjectives(summarized_text)

#     return simplified_text

# import os
# import fitz  # PyMuPDF
# import numpy as np
# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# def extract_text_from_pdfs(pdf_folder):
#     text_data = {}
#     for filename in os.listdir(pdf_folder):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(pdf_folder, filename)
#             pdf_document = fitz.open(pdf_path)
#             text = ""
#             for page in pdf_document:
#                 text += page.get_text()
#             text_data[filename] = text
#     return text_data

# def summarize_text(text, model, tokenizer, max_length=150):
#     inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# def main(pdf_folder, query):
#     # Load T5 model and tokenizer
#     model_name = "t5-small"
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)

#     # Extract text from PDFs
#     pdf_texts = extract_text_from_pdfs(pdf_folder)

#     # Summarize extracted text
#     summaries = {}
#     for filename, text in pdf_texts.items():
#         summarized_text = summarize_text(text, model, tokenizer)
#         summaries[filename] = summarized_text

#     # Here you can implement the logic to retrieve summaries based on the query
#     return summaries

# if __name__ == "__main__":
#     pdf_folder = "./pdfs"
#     query = "Wie hoch ist die Grundzulage?"
#     summaries = main(pdf_folder, query)
#     for filename, summary in summaries.items():
#         print(f"Summary for {filename}:")
#         print(summary)



# if __name__ == "__main__":
#     import os
#     pdf_folder = "./pdfs"
#     query = "Wie hoch ist die Grundzulage?"
#     response = main(pdf_folder, query)
#     print("Question: ", query, ":\n\n", response)


# import fitz  # PyMuPDF
# import os
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline
# import spacy
# import torch

# # Step 1: Initialize the model for text embedding and summarization
# embedding_model = SentenceTransformer('distilbert-base-german-cased') # ('distilbert-base-nli-mean-tokens')
# # Check if GPU is available and set device accordingly
# device = 0 if torch.cuda.is_available() else -1  # Use 0 for the first GPU, -1 for CPU
# summarization_model = pipeline("summarization", model="t5-base", tokenizer="t5-base", device=device)

# # Load the German language model in spaCy
# nlp = spacy.load("de_core_news_sm")


# # Step 2: Function to extract text from PDFs
# def extract_text_from_pdfs(pdf_directory):
#     text_data = []
#     for filename in os.listdir(pdf_directory):
#         if filename.endswith('.pdf'):
#             pdf_path = os.path.join(pdf_directory, filename)
#             doc = fitz.open(pdf_path)
#             for page in doc:
#                 text = page.get_text()
#                 text_data.append(text)
#             doc.close()
#     return text_data

# # Step 3: Break down text into paragraphs
# def break_into_paragraphs(text):
#     # paragraphs = []
#     # for txt in text:
#     #     paragraphs.extend(txt.split('\n\n'))  # Split based on double new lines for paragraphs
#     # return paragraphs
#     doc = nlp(text)
#     sections = []
#     current_section = []

#     for sent in doc.sents:
#         current_section.append(sent.text)
#         # Check for section delimiters like headings or bullet points
#         if sent.text.strip().endswith(':') or sent.text.strip().startswith('-'):
#             sections.append(" ".join(current_section))
#             current_section = []

#     if current_section:
#         sections.append(" ".join(current_section))

#     return sections

# # Step 4: Create embeddings and store in a vector database
# def store_in_vector_database(paragraphs):
#     embeddings = embedding_model.encode(paragraphs)
#     dim = embeddings.shape[1]
    
#     # Create a FAISS index
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings.astype(np.float32))  # Add embeddings to the index
#     return index, paragraphs

# # Step 5: Function to retrieve relevant passages
# def retrieve_relevant_passages(query, index, paragraphs, k=10):
#     query_embedding = embedding_model.encode([query]).astype(np.float32)
#     _, indices = index.search(query_embedding, k)
#     return [paragraphs[i] for i in indices[0]]

# # Step 6: Create a coherent response with summarization
# def create_response(paragraphs):
#     if paragraphs:
#         # Summarize retrieved paragraphs
#         combined_text = " ".join(paragraphs)
#         summarized_response = summarization_model(combined_text, max_length=500, min_length=150, do_sample=False)
#         return summarized_response[0]['summary_text']
#     return "No relevant information found."

# # Main script execution
# if __name__ == "__main__":
#     pdf_directory = './pdfs'  # Replace with your PDF directory
#     text_data = extract_text_from_pdfs(pdf_directory)

#      # Split text into meaningful sections
#     paragraphs = []
#     for text in text_data:
#         paragraphs.extend(break_into_paragraphs(text))
#     # paragraphs = break_into_paragraphs(text_data)
    
#     index, stored_paragraphs = store_in_vector_database(paragraphs)

#     # queries = [
#     #     "Wie hoch ist die Grundzulage?",
#     #     "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
#     #     "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
#     # ]
    
#     # for query in queries:
#     #     relevant_passages = retrieve_relevant_passages(query, index, stored_paragraphs)
#     #     response = create_response(relevant_passages)
#     #     print(f"Query: {query}\nResponse: {response}\n")

#     # query = "Wie hoch ist die Grundzulage?"
#     query = "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?"
#     relevant_passages = retrieve_relevant_passages(query, index, stored_paragraphs)
#     response = create_response(relevant_passages)
#     print(f"Query: {query}\nResponse: {response}\n")    


# import spacy

# Load German model
# nlp = spacy.load("de_core_news_sm")



import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

def extract_text_from_pdf(pdf_path):
    text_sections = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            sections = text.split("\n\n")  # Split into paragraphs
            text_sections.extend(sections)
    return text_sections

# Extract text from all PDF files in a directory
def extract_texts_from_pdfs(directory):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text_sections = extract_text_from_pdf(pdf_path)
            all_texts.extend(text_sections)
    return all_texts



# Load a pre-trained model for creating embeddings (e.g., multilingual model)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def create_embeddings(text_sections):
    embeddings = model.encode(text_sections)
    return np.array(embeddings)

def store_embeddings_in_faiss(embeddings, text_sections):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(embeddings)  # Add vectors to the index
    return index, text_sections

# Combine extraction and storage steps
def process_and_store_texts(directory):
    text_sections = extract_texts_from_pdfs(directory)
    embeddings = create_embeddings(text_sections)
    index, texts = store_embeddings_in_faiss(embeddings, text_sections)
    return index, texts


def get_relevant_passages(query, index, texts, k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k)
    results = [texts[idx] for idx in indices[0]]
    return results



# Load a T5 model fine-tuned for summarization
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
import torch 

def generate_response(passages):
    combined_text = " ".join(passages)
    input_text = "summarize: " + combined_text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():                 # Disable gradient calculation to speed up the process
        summary_ids = t5_model.generate(
            input_ids, 
            max_length=150,               # Increase this to make the response longer
            min_length=50,                # Set a minimum length to avoid very short summaries
            length_penalty=1.0,           # Lower values like 0.5 can produce longer texts
            num_beams=4, 
            early_stopping=True
        )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    directory = './pdfs'
    index, texts = process_and_store_texts(directory)
    
    # Example query
    # query = "Wie hoch ist die Grundzulage?"
    # query = "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?"
    query = "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    passages = get_relevant_passages(query, index, texts)
    response = generate_response(passages)
    
    print("Frage:", query, ":\n\n", response)
    print(response)

if __name__ == "__main__":
    main()
