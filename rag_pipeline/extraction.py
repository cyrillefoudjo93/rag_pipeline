import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text_sections = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            sections = text.split("\n\n")  # Split into paragraphs
            text_sections.extend(sections)
    return text_sections

def extract_texts_from_pdfs(directory):
    import os
    all_texts = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text_sections = extract_text_from_pdf(pdf_path)
            all_texts.extend(text_sections)
    return all_texts
