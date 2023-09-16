"""Calculate tf-idf for words in a set of documents and calculate pairwise cosine similarity for the documents"""
import os
from pathlib import Path
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_data(dir: str):
    """Read all text files in the given directory.

    Args:
        dir (str): The directory to read the text files from.

    Returns:
        dict: A dictionary with the file name as the key and the file contents as the value.
    """
    files = list(Path(dir).glob("*.txt"))
    data = {}
    for f in files:
        with open(f, "r", encoding="UTF-8") as file:
            data[f.stem] = file.read()
            print(f"Read file: {f}")
    return data


def write_data(filename, data, format_string):
    with open(filename, "w", encoding="UTF-8") as file:
        for k, v in data.items():
            file.write(format_string.format(k, v))
            file.write("\n-------------------\n\n")


def tokenize_words(data: dict):
    """Tokenize the words in the given data."""
    tokenized_data = {}
    for k, v in data.items():
        word_tokens = word_tokenize(v)
        tokenized_data[k] = word_tokens

    write_data("tokenized_words.txt", tokenized_data, "Tokenized words for {}:\n{}")
    return tokenized_data


def remove_stop_words(data: dict):
    """Remove stop words from the given data."""
    stop_words = set(stopwords.words("english")) | set(string.punctuation)
    no_stop_words_data = {}
    for k, v in data.items():
        no_stop_words = [w for w in v if not w in stop_words]
        no_stop_words_data[k] = no_stop_words

    write_data("no_stop_words.txt", no_stop_words_data, "Removed stop words for {}:\n{}")
    return no_stop_words_data


def main():
    # Uncomment and run this line once to download the nltk data
    # nltk.download()

    data = read_data("data")

    # Prepare to work with the documents
    # At this time, the only preprocessing step is to lowercase the documents
    data = {k: v.lower() for k, v in data.items()}

    # Task 1 - tokenize words
    tokenized = tokenize_words(data)

    # Task 1 - remove stop words
    remove_stop_words(tokenized)

    # Task 1 - stem

    # Task 2:
    # - Calculate tf-idf for each word in each document and generate document-word
    #   matrix (each element in the matrix is the tf-idf score for a word in a document)

    # Task 3:
    # - Calculate pairwise cosine similarity for the documents



if __name__ == "__main__":
    main()
