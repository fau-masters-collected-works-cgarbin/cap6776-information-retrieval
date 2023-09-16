# Information Retrieval class

Assignments and notes for the information retrieval class.

If you just cloned the repository, please read the [development environment](#development-environment) section before proceeding.

## Assignment 1 - Basic NLP pipeline with NLTK and scikit-learn

### Assignment description

This assignment is the basic NLP pipeline using [NLTK](https://www.nltk.org/) and [scikit-learn](https://scikit-learn.org/stable/).

- Tokenization
- Stop words removal
- Stemming
- TF-IDF calculation
- Pairwise cosine similarity calculation

This is the description, copied verbatim from the assignment.

_NOTE: Follow the instructions in the [development environment](#development-environment) section to set up the environment. The instructions below (from the assignment) are for reference only. They are missing some dependencies (e.g. scikit-learn) and do not specify the version to install (code may break in the future)._

> Given a collection of documents, conduct text preprocessing including tokenization, stop words removal, stemming, tf-idf calculation, and pairwise cosine similarity calculation using NLTK. The following steps should be completed:
>
> - Install Python and NLTK
> - Tokenize the documents into words, remove stop words, and conduct stemming
> - Calculate tf-idf for each word in each document and generate document-word matrix (each element in the matrix is the tf-idf score for a word in a document)
> - Calculate pairwise cosine similarity for the documents

Run the assignment (configure the [development environment](#development-environment) if you haven't done so yet):

```bash
source venv/bin/activate
cd assignment1-nltk
python tf-idf-doc-matrix.py
```

## Development environment

Create a Python virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
