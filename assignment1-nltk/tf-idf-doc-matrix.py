"""Calculate tf-idf for words in a set of documents and calculate pairwise cosine similarity for the documents"""
import os
import string
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def download_nltk_data():
    """Download necessary NLTK data files if not already present."""
    necessary_data_packages = [
        ("corpora/stopwords", "stopwords"),
        ("tokenizers/punkt", "punkt")
    ]

    for data_path, package in necessary_data_packages:
        try:
            nltk.data.find(data_path)
        except LookupError:
            nltk.download(package)


def read_data(dir: str):
    """Read all text files in the given directory."""
    files = list(Path(dir).glob("*.txt"))
    data = {}
    for f in files:
        with open(f, "r", encoding="UTF-8") as file:
            data[f.stem] = file.read()
    return data


def full_path_from_name(name: str):
    """Return the path into a results directory to keep the results file separate."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = results_dir / name
    return str(filename)


def write_data(filename, data, format_string):
    filename = full_path_from_name(filename)
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

    write_data("task1.1_tokenized_words.txt", tokenized_data, "Tokenized words for {}:\n{}")
    return tokenized_data


def remove_stop_words(data: dict):
    """Remove stop words from the given data."""
    stop_words = set(stopwords.words("english")) | set(string.punctuation)
    no_stop_words_data = {}
    for k, v in data.items():
        no_stop_words = [w for w in v if not w in stop_words]
        no_stop_words_data[k] = no_stop_words

    write_data("task1.2_no_stop_words.txt", no_stop_words_data, "Removed stop words for {}:\n{}")
    return no_stop_words_data


def stem_words(data: dict):
    """Stem the words in the given data."""
    stemmer = PorterStemmer()
    stemmed_data = {}
    for k, v in data.items():
        stemmed_words = [stemmer.stem(w) for w in v]
        stemmed_data[k] = stemmed_words

    write_data("task1.3_stemmed_words.txt", stemmed_data, "Stemmed words for {}:\n{}")
    return stemmed_data


def calculate_tf_idf(stemmed_text: dict):
    """Calculate tf-idf for the given data."""
    vectorizer = TfidfVectorizer()
    tf_idfs = vectorizer.fit_transform(stemmed_text.values())
    doc_matrix = tf_idfs.toarray()
    set_vocab = vectorizer.get_feature_names_out()

    with open(full_path_from_name("task2_tf_idf.txt"), "w", encoding="UTF-8") as file:
        # Add the header
        header = f"{'Word':<20}" + "  ".join([f"{file_name:<17}" for file_name in stemmed_text.keys()])
        file.write(header + "\n")

        # Print the TF-IDF values for each term in each file_content entry
        # The order of the files here is the same as in the header because starting in Python 3.7 dictionaries
        # remember the order of insertion
        row_format = f"{{:<20}}" + "  ".join([f"{{:<.15f}}" for _ in range(len(stemmed_text))])
        for i, word in enumerate(set_vocab):
            row = row_format.format(word, *doc_matrix[:, i])
            file.write(row + "\n")

    return tf_idfs


def calculate_cosine_similarity(tf_idfs, doc_names: list):
    """Calculate pairwise cosine similarity for the given data.

    Args:
        tfs: TF-IDF matrix
        doc_names: List of document names, in the same order as the rows in the TF-IDF matrix
    """
    with open(full_path_from_name("task3_cosine_similarity.txt"), "w", encoding="UTF-8") as file:
        for i in range(len(doc_names)):
            for j in range(i+1, len(doc_names)):
                similarity = cosine_similarity(tf_idfs[i], tf_idfs[j])
                file.write(f"Similarity between {doc_names[i]} and {doc_names[j]}: {similarity}\n")
                print(f"Similarity between {doc_names[i]} and {doc_names[j]}: {similarity}\n")


def main():
    download_nltk_data()

    print("Reading data...")
    file_contents = read_data("data")

    # Prepare to work with the documents
    # At this time, the only preprocessing step is to lowercase the documents
    file_contents = {k: v.lower() for k, v in file_contents.items()}

    # Task 1.1
    # - Tokenize the documents into words...
    print("Tokenizing words...")
    tokenized = tokenize_words(file_contents)

    # Task 1.2
    # - ...remove stop words
    print("Removing stop words...")
    no_stop_words = remove_stop_words(tokenized)

    # Task 1.3
    # - ...and conduct stemming
    print("Stemming words...")
    stemmed_words = stem_words(no_stop_words)

    # Task 2:
    # - Calculate tf-idf for each word in each document and generate document-word
    #   matrix (each element in the matrix is the tf-idf score for a word in a document)
    # Note that we train the TF-IDF vectorizer on fully preprocessed text (tokenized, removed stop words, stemmed)
    # See https://courses.cs.duke.edu/spring14/compsci290/assignments/lab02.html for an example
    # Note that in the example it uses the `tokenizer` parameter to pass a custom tokenizer, while here we use
    # a preprocessed text, so we don't need to pass a custom tokenizer
    # Similar example in https://saturncloud.io/blog/how-to-pass-a-preprocessor-to-tfidfvectorizer-in-python/
    # Prepare the data for the TF-IDF calculation
    stemmed_as_strings = {k: " ".join(v) for k, v in stemmed_words.items()}
    print("Calculating TF-IDF...")
    tf_idfs = calculate_tf_idf(stemmed_as_strings)

    # Task 3:
    # - Calculate pairwise cosine similarity for the documents
    print("Calculating cosine similarity...")
    calculate_cosine_similarity(tf_idfs, list(file_contents.keys()))

    print("\nDone. Results in these files:")
    all_files = sorted(Path("results").glob("*"))
    for f in all_files:
        print(f)



if __name__ == "__main__":
    # In case it was started from the debugger (runs from the top-level directory)
    # The code in the script assumes it is run from the assignment1-nltk directory
    dir = "assignment1-nltk"
    if Path.cwd().name != dir:
        os.chdir(dir)

    main()
