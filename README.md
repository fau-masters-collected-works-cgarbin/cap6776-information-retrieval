# Information Retrieval class

Assignments and notes for the FAU's CAP-6776 information retrieval class.

If you just cloned the repository, please read the [development environment](#development-environment) section before proceeding.

## Assignment 1 - Basic NLP pipeline with NLTK and scikit-learn

### Assignment description

This assignment is a basic NLP pipeline using [NLTK](https://www.nltk.org/) and [scikit-learn](https://scikit-learn.org/stable/).

- Tokenization
- Stop words removal
- Stemming
- TF-IDF calculation
- Pairwise cosine similarity calculation

Assignment description:

_NOTE: Follow the instructions in the [development environment](#development-environment) section to set up the environment. The instructions below (from the assignment) are for reference only. They are missing some dependencies (e.g. scikit-learn) and do not specify the version to install (code may break in the future)._

> Given a collection of documents, conduct text preprocessing including tokenization, stop words removal, stemming, tf-idf calculation, and pairwise cosine similarity calculation using NLTK. The following steps should be completed:
>
> - Install Python and NLTK
> - Tokenize the documents into words, remove stop words, and conduct stemming
> - Calculate tf-idf for each word in each document and generate document-word matrix (each element in the matrix is the tf-idf score for a word in a document)
> - Calculate pairwise cosine similarity for the documents

To run the assignment (configure the [development environment](#development-environment) if you haven't done so yet):

```bash
source venv/bin/activate
cd assignment1-nltk
python tf-idf-doc-matrix.py
```

## Assignment 2 - Class project

See the [project that summarizes GitHub issues with large language models (LLM)](https://github.com/fau-masters-collected-works-cgarbin/llm-github-issues).

## Assignment 3 - Image classification using TensorFlow Mobilenet

Image classification with [TensorFlow Mobilenet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet).

To run the assignment (configure the [development environment](#development-environment) if you haven't done so yet):

```bash
source venv/bin/activate
cd assignment3-image-classification
python image-classification.py
```

## Development environment

Create a Python virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
