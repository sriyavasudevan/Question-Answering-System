
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_time(fun, *args, **kwargs):
    """
    Gets the runtime of a method
    :param fun: name of the method
    :param args: variable arguments
    :param kwargs: variable arguments
    :return: time in seconds
    """
    start = time()
    res = fun(*args, **kwargs)
    return time() - start, res

'''
def segment_file(file, max_file_length=450):
    """
    Separate the file to make sure max file length is under BERT limit
    :param file: current file used
    :param max_file_length: settled maximum file limit
    :return: the split file
    """
    # List containing full and segmented docs
    segmented_file = []

    for file in file:
        # Split document by spaces to obtain a word count that roughly approximates the token count
        split_to_words = file.split(" ")

        # If the document is longer than our maximum length, split it up into smaller segments and add them to the list
        if len(split_to_words) > max_file_length:
            for file_segment in range(0, len(split_to_words), max_file_length):
                segmented_file.append(" ".join(split_to_words[file_segment:file_segment + max_file_length]))

        # If the document is shorter than our maximum length, add it to the list
        else:
            segmented_file.append(file)

    return segmented_file
'''

def get_top_k_file(question, file_df, k=2):
    """
    Apply TF-IDF method to get top k related file according to the question
    :param question: the input question being asked
    :param file_df: current collection of files
    :param k: the number of returned files
    :return: top k related file
    """
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    # convert the df into a file
    file_list = file_df['text'].to_list()

    # Add the question to the front of the list
    question_and_file = [question] + file_list

    # create a TF-IDF matrix between each document and the vocab
    matrix = vectorizer.fit_transform(question_and_file)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    query_text_vectorized = matrix[0]
    for i in range(1, len(question_and_file)):
        cos_sim_matrix = cosine_similarity(matrix[i], query_text_vectorized)
        cos_sim = cos_sim_matrix[0][0]
        scores.append(cos_sim)

    # sort the list in descending order after enumerating it
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_file_indices = [x[0] for x in sorted_list[:k]]

    # the file with the highest score
    top_file = [file_list[x] for x in top_file_indices]

    return top_file