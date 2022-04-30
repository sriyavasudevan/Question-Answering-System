import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from time import time
import platform
import file_io


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


def get_tokens_helper(*args):
    """
    a helper method that will get the tokens for the text passed in
    :param args: variable argument where either question and context OR just context will be passed
    :return: inputs, input_ids and text_tokens corresponding to the input context
    """
    if len(args) == 2:
        question = args[0]
        context = args[1]
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    elif len(args) == 1:
        to_be_tokenized = args[0]
        inputs = tokenizer.encode_plus(to_be_tokenized, add_special_tokens=True, return_tensors="pt")
    else:
        print("error in tokenization")

    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return inputs, input_ids, text_tokens


def get_answer_helper(answer_start, answer_end, input_ids, text_tokens):
    """
    a helper method that gets the answer based on the answer starting and ending indices, also checks
    for any ## and formats the answer nicely
    :param answer_start: starting index of answer
    :param answer_end: ending index of answer
    :param input_ids: input ids from the context
    :param text_tokens: tokenized version of the text sent into BERT
    :return: answer in string format
    """
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    if answer_end >= answer_start:
        answer = text_tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if text_tokens[i][0:2] == "##":
                answer += text_tokens[i][2:]
            else:
                answer += " " + text_tokens[i]

    return answer


def map_question_to_num_tokens(question, map):
    """
    a helper method to map the question to number of tokens in that question
    :param question: the question to be tokenized
    :param map: current map of question to number of tokens
    :return: map of question to number of tokens
    """

    # need to get the number of tokens in each question
    question_inputs, question_input_ids, question_text_tokens = get_tokens_helper(question)

    # key - question and value - number of tokens in the current question
    map[question] = len(question_text_tokens)

    return map


def question_answer(question, context, question_history, map):
    """
    This is the main code for qa system - uses BERT
    :param question: current question being asked
    :param context: current context used
    :param question_history: list of questions previously asked
    :param map: current map of question to number of tokens
    :return: current context, current question history, predicted answer
    """

    # using a helper method to get tokens
    inputs, input_ids, text_tokens = get_tokens_helper(question, context)

    """ Using below line to test out removing entire questions from question history """
    map = map_question_to_num_tokens(question, map)

    # sending tokens to BERT model
    model_out = model(**inputs)

    # getting starting ending indices for predicted answer
    answer_start = torch.argmax(model_out.start_logits)
    answer_end = torch.argmax(model_out.end_logits)

    # using a helper method to check for the answer
    answer = get_answer_helper(answer_start, answer_end, input_ids, text_tokens)

    # checking for a null answer
    if answer.startswith("[CLS]") or answer == " ":
        answer = "Unable to find the answer to your question."
    print("\nPredicted answer:\n{}".format(answer.capitalize()))

    # strings of question_history and the full context
    question_history_str = ''.join(question_history)
    # all_text = ''.join(question_history) + context
    context_str = ''.join(question_history) + original_context

    # finding number of tokens in question_history and the full context
    all_text_inputs, all_text_input_ids, all_text_text_tokens = get_tokens_helper(context_str)
    question_history_inputs, question_history_input_ids, question_history_text_tokens = get_tokens_helper(
        question_history_str)

    # checking if the length of the context tokens is over 512, since BERT has a restriction
    if len(all_text_text_tokens) <= BERT_TOKEN_LIMIT:
        context = context_str
    else:

        # tokenized version of the original plain context without question history
        original_context_t = get_tokens_helper(original_context)
        original_context_tokens = original_context_t[2]
        # original_context_tokens.pop(0) # removing CLS token

        # how many tokens are over limit
        tokens_over_limit = len(all_text_text_tokens) - BERT_TOKEN_LIMIT

        """
        # Old code block to remove tokens over limit instead of whole questions
        
        # tokens to be kept in question history
        tokens_to_keep = abs(len(question_history_text_tokens) - tokens_over_limit - 1)

        # truncated question history tokens + removing CLS token in the front
        question_history_text_tokens = question_history_text_tokens[1:tokens_to_keep]

        # add the new question history back to the original context
        truncated_all_context_tokens = question_history_text_tokens + original_context_tokens

        # convert tokens back to proper string
        new_truncated_context = tokenizer.convert_tokens_to_string(truncated_all_context_tokens)

        context = new_truncated_context
        """

        # below code checks how many questions needs to be removed based on the amount of tokens over limit
        count = 0
        total_tokens = 0
        for i in range(0, len(question_history)):
            current_q = question_history[i].strip(' ')
            current_num = map.get(current_q)
            total_tokens += current_num
            if total_tokens < tokens_over_limit:
                count += 1
                continue
            else:
                break

        # removes questions from question history
        question_history = question_history[count + 1:]

        new_truncated_history_str = ''.join(question_history)
        new_truncated_context_with_question_history = new_truncated_history_str + original_context
        context = new_truncated_context_with_question_history

    print("Current context: " + context)
    return context, question_history, answer


def test_conversation(initial_context, question_list, map):
    """
    Can use this method to directly pass in a list of questions instead of typing it in
    :param initial_context: the initial piece of context
    :param question_list: list of questions you want to ask
    :param map current map of question to number of tokens
    :return: nothing
    """
    question_history = []
    logging_list = []
    text = initial_context

    for i, question in enumerate(question_list):
        """if question + " " not in question_history:
            question_history.append(question + " ")"""
        question_history.append(question + " ")
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, question_history, map)
        text = qa_returned_elements[0]
        question_history = qa_returned_elements[1]
        current_answer = qa_returned_elements[2]

        # logging
        logging_list.append([question, ''.join(question_history), current_answer, str(time_taken_to_answer)])
        logging_df = pd.DataFrame(logging_list,
                                  columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
        if i == 0:
            file_io.write_data(logging_df, first_time=True)
        else:
            file_io.write_data(logging_df)
        logging_list = []


def begin_conversation(initial_context, map):
    """
    This method output instructions to begin the conversation
    :param initial_context: the inital piece of text
    :param map current map of question to number of tokens
    :return: nothing
    """
    question_history = []
    # logging_list =[]

    question = input("\nPlease enter your question: \n")
    question_history.append(question + " ")
    text = initial_context

    while True:
        # checking the time taken by each call
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, question_history, map)
        print("Time taken: " + str(time_taken_to_answer))
        current_answer = qa_returned_elements[1]

        """logging
        logging_list.append([question, ''.join(question_history), current_answer, str(time_taken_to_answer)])
        logging_df = pd.DataFrame(logging_list, columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
        file_io.write_data(logging_df)
        logging_list = []"""

        flag = True
        flag_N = False

        while flag:
            response = input("\nDo you want to ask another question based on this text (Y/N)? ")
            if response[0] == "Y":
                question = input("\nPlease enter your question: \n")
                question_history.append(question + " ")
                flag = False

            elif response[0] == "N":
                print("\nBye!")
                flag = False
                flag_N = True

        if flag_N:
            break


# initialize the model and the tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
BERT_TOKEN_LIMIT = 512
map_qns_to_num_tokens = {}

# checking the current environment type
print("Current environment: ")
print(platform.uname())

"""
# pick the context from dataset
index = 11
original_context = file_io.read_data("sample_dataset.json")["text"][index]

# print the initial context and start conversation
print("The initial context: " + original_context)
begin_conversation(original_context, map_qns_to_num_tokens)
"""

# use this block of code to be able to feed questions directly instead of typing each time
# data file contains text and list of questions
test_df = file_io.read_data('test_file_to_show.json')

# for i in range(0, test_df.shape[0]):
original_context = test_df["data"][5]["text"]
q_list = test_df["data"][5]["questions"]
test_conversation(original_context, q_list, map_qns_to_num_tokens)
