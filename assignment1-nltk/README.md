# Assignment 1 - NLTK

## Assignment description

Given a collection of documents, conduct text preprocessing including tokenization, stop words removal, stemming, tf-idf calculation, and pairwise cosine similarity calculation using NLTK. The following steps should be completed:

- Install Python and NLTK
- Tokenize the documents into words, remove stop words, and conduct stemming
- Calculate tf-idf for each word in each document and generate document-word matrix (each element in the matrix is the tf-idf score for a word in a document)
- Calculate pairwise cosine similarity for the documents

Please include your screen shots for each of the above steps and also the final results of the pairwise cosine similarity scores in your report.

## Development environment

Create a Python virtual environment and install NLTK with the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
