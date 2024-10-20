import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'Page \d+|\d+ / \d+', '', text)  # Remove page numbers
    return text.strip()
