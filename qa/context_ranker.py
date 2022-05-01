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


def get_top_k_file(query, file_df, k=2):
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    file_list = file_df['text'].to_list()

    # Create a corpus of query and documents and convert to TFIDF vectors
    query_and_file = [query] + file_df
    list_q_file = query_and_file['text'].to_list()
    matrix = vectorizer.fit_transform(list_q_file)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    for i in range(1, len(list_q_file)):
        scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # enum_obj = enumerate(scores)
    # test_list = list(enum_obj)
    # Sort list of scores and return the top k highest scoring documents
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_file_indices = [x[0] for x in sorted_list[:k]]
    top_file = [file_list[x] for x in top_file_indices]

    return top_file


def question_answer(question, text):
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


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

text_df = pd.read_json('ranker_text_file.json')

# question = input("\nPlease enter your question: \n")
question = "What are the trip wires?"

text = text_df
candidate_file = get_top_k_file(question, text, 3)

while True:
    for i in candidate_file:
        question_answer(question, i)
        print("Reference Document: ", i)
        print("/")

        flag = True
        flag_N = False

        while flag:
            response = input("\nDo you want to ask another question based on this text (Y/N)? ")
            if response[0] == "Y":
                question = input("\nPlease enter your question: \n")
                flag = False
            elif response[0] == "N":
                print("\nBye!")
                flag = False
                flag_N = True

            if flag_N:
                break
