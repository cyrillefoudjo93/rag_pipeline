# RAG Pipeline

This Python package implements a Retrieval-Augmented Generation (RAG) pipeline for extracting text from PDFs, storing it in a vector database, and querying for relevant information.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/cyrillefoudjo93/rag_pipeline.git
cd rag_pipeline
pip install -e .
```

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage
Run the pipeline with:
```bash
python main.py
```
If it dosen't work, try to run following script (it contains all necessary functions used on the RAG Pipeline) from the dev directory:
```bash
cd dev
python tryIt.py
``` 

## Dependecies
- PyMuPDF
- sentence-transformers
- faiss-cpu
- transformers
- torch
- numpy
- sentencepiece
- protobuf