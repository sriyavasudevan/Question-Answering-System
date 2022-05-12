import pandas as pd
from time import time

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

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


def question_answer(question, text):
    """
    Main method for qa system with BERT model (test)
    :param question: current question being asked
    :param text: current text used
    :return: predicted answer
    """
    # tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    # number of tokens in segment A - question
    num_seg_a = sep_idx + 1

    # number of tokens in segment B - text
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)

    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    #     print("Text:\n{}".format(text.capitalize()))
    #     print("\nQuestion:\n{}".format(question.capitalize()))
    print("\nAnswer:\n{}".format(answer.capitalize()))


# initialize the model and the tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# loading the text
text_df = pd.read_json('ranker_text_file.json')
# set the input question
question = input("\nPlease enter your question: \n")

# this while loop just finds out which documents are in the top 3

while True:

    # candidate_file = get_top_k_file(question, text_df, 3)
    time_taken_to_rank, candidate_file = get_time(get_top_k_file, question, text_df, 3)

    # for i in candidate_file:

    # question_answer(question, candidate_file)
    print("Top 3 Reference Document:\n", candidate_file[0] + "\n" + candidate_file[1]
          + "\n" + candidate_file[2])

    # time taken to rank the context
    print("Time taken to rank: " + str(time_taken_to_rank))

    flag = True
    flag_N = False

    while flag:
        response = input("\nDo you want to ask another question based on this text (Y/N)? ")
        if response[0] == "Y":
            question = input("\nPlease enter your question: \n")
            # history.append(question + " ")
            flag = False

        elif response[0] == "N":
            print("\nBye!")
            flag = False
            flag_N = True

    if flag_N:
        break
