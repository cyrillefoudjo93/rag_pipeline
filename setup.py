from setuptools import setup, find_packages

setup(
    name='rag_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'PyMuPDF',
        'sentence-transformers',
        'faiss-cpu',
        'transformers',
        'torch',
        'numpy'
    ],
    description='A RAG pipeline for extracting, storing, and querying text from PDF documents.',
    author='Cyrille Konzeu',
    author_email='cyrillekonzeu@gmail.com',
    url='https://github.com/cyrillefoudjo93/rag_pipeline',
)
