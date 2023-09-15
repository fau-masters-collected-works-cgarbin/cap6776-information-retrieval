"""Calculate tf-idf for words in a set of documents and calculate pairwise cosine similarity for the documents"""
import os
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Read all data files
def read_data():
    # List all .txt files in the data directory
    # Use pathlib to search files, do not use os module
    files = list(Path("data").glob("*.txt"))
    # Read each file and store the contents in a list for file
    data = []
    for f in files:
        # Use pathlib to OS agnostic path
        with open(f, "r", encoding="UTF-8") as file:
            data.append(file.read())
            print(f"Read file: {f}")
    return data


def main():
    data = read_data()


if __name__ == "__main__":
    main()
