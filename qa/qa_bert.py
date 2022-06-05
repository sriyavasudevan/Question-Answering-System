import pandas as pd
# import numpy as np
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


def get_confidence(start_logits, end_logits):
    """
    This method will determine the confidence of the answer based on a threshold probability
    :param start_logits: start logits
    :param end_logits: end logits
    :return: confident true or false
    """
    prob_dist_start = torch.round(torch.softmax(start_logits, dim=1), decimals=3)
    prob_dist_end = torch.round(torch.softmax(end_logits, dim=1), decimals=3)

    arg_start_index = torch.argmax(start_logits)
    arg_end_index = torch.argmax(end_logits)

    softm_start_index = torch.argmax(prob_dist_start)
    softm_end_index = torch.argmax(prob_dist_end)

    if arg_start_index == softm_start_index and arg_end_index == softm_end_index:

        start_prob = prob_dist_start[0, softm_start_index]
        end_prob = prob_dist_end[0, softm_end_index]

        if start_prob >= 0.70 and end_prob >= 0.70:
            confident = True
        else:
            confident = False

    return confident, start_prob, end_prob


def get_slide_number(context):
    """
    This method can help to get slide number from input context
    :param context: input original context
    """
    split_content = context.split(r' ')
    slide_number = ' ' + 'Slide' + ' ' + split_content[1]
    return slide_number


def question_answer(question, context, history, map):
    """
    This is the main code for qa system - uses BERT
    :param question: current question being asked
    :param context: current context used
    :param history: list of questions previously asked
    :param map: current map of question to number of tokens
    :return: current context, current question history, predicted answer
    """
    # checking if the question is end with a question mark
    last_character = question[-1]
    if last_character != "?":
        question = question + "?" 
    
    # using a helper method to get tokens
    inputs, input_ids, text_tokens = get_tokens_helper(question, context)

    # sending tokens to BERT model
    model_out = model(**inputs)

    # getting starting ending indices for predicted answer
    answer_start = torch.argmax(model_out.start_logits)
    answer_end = torch.argmax(model_out.end_logits)

    # check the confidence of the answer, returns a boolean based on threshold
    confidence_components = get_confidence(model_out.start_logits, model_out.end_logits)
    start_prob = confidence_components[1]
    end_prob = confidence_components[2]

    """if not confident:
        answer = ""
    else:
        """
    # using a helper method to check for the answer
    answer = get_answer_helper(answer_start, answer_end, input_ids, text_tokens)

    # checking for a null answer
    if answer.startswith("[CLS]") or answer == " " or answer == "":
        answer = "Unable to find the answer to your question."
        
    # checking if there is a period at the end of the answer
    last_answer = answer[-1]
    if last_answer != ".":
        answer = answer + "."
                
    print("\nPredicted answer:\n{}".format(answer.capitalize()))
    print("Confidence start score: " + str(start_prob) + ", end score: " + str(end_prob))
    print("For more information, please go to" + get_slide_number(original_context))

    # strings of history and the full context
    history_component = question + " "
    history.append(history_component)

    """ Using below line to test out removing entire q&a from history """
    map = map_question_to_num_tokens(history_component, map)

    context_str = ''.join(history) + original_context

    # finding number of tokens in history and the full context
    all_text_inputs, all_text_input_ids, all_text_text_tokens = get_tokens_helper(context_str)

    # checking if the length of the context tokens is over 512, since BERT has a restriction
    if len(all_text_text_tokens) < BERT_TOKEN_LIMIT:
        context = context_str
    else:

        # tokenized version of the original plain context without question history
        original_context_t = get_tokens_helper(original_context)

        # how many tokens are over limit
        tokens_over_limit = len(all_text_text_tokens) - BERT_TOKEN_LIMIT

        # below code checks how many questions needs to be removed based on the amount of tokens over limit
        count = 0
        total_tokens = 0
        for i in range(0, len(history)):
            # current_history_element = history[i].strip(' ')
            current_history_element = history[i]
            current_num = map.get(current_history_element)
            total_tokens += current_num
            if total_tokens < tokens_over_limit:
                count += 1
                continue
            else:
                break

        # removes questions from question history
        history = history[count + 1:]

        new_truncated_history_str = ''.join(history)
        new_truncated_context_with_history = new_truncated_history_str + original_context
        context = new_truncated_context_with_history

    print("Current context: " + context)
    return context, history, answer


def test_conversation(initial_context, question_list, map):
    """
    Can use this method to directly pass in a list of questions instead of typing it in
    :param initial_context: the initial piece of context
    :param question_list: list of questions you want to ask
    :param map current map of question to number of tokens
    :return: nothing
    """
    history = []
    logging_list = []
    text = initial_context

    for i, question in enumerate(question_list):
        """if question + " " not in history:
            history.append(question + " ")"""
        # history.append(question + " ")
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, history, map)
        text = qa_returned_elements[0]
        history = qa_returned_elements[1]
        current_answer = qa_returned_elements[2]

        # logging
        logging_list.append([question, ''.join(history), current_answer, str(time_taken_to_answer)])
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
    history = []
    # logging_list =[]

    question = input("\nPlease enter your question: \n")
    text = initial_context

    while True:
        # checking the timcurrent_history_element = history[i]e taken by each call
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, history, map)
        print("Time taken: " + str(time_taken_to_answer))
        text = qa_returned_elements[0]
        history = qa_returned_elements[1]
        current_answer = qa_returned_elements[2]

        """logging
        logging_list.append([question, ''.join(history), current_answer, str(time_taken_to_answer)])
        logging_df = pd.DataFrame(logging_list, columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
        file_io.write_data(logging_df)
        logging_list = []"""

        flag = True
        flag_N = False

        while flag:
            response = input("\nDo you want to ask another question based on this text (Y/N)? ")
            if response[0] == "Y":
                question = input("\nPlease enter your question: \n")
                history.append(question + " ")
                flag = False

            elif response[0] == "N":
                print("\nBye!")
                flag = False
                flag_N = True

        if flag_N:
            break

def inset_links(answer):
    import re
    ans_for_display = answer

    df = pd.read_csv('map_of_hyperlinks.csv')

    list = re.findall("link\d\d", ans_for_display)

    for link in list:
        hyperlink = df.loc[df['Link'] == link]['Hyperlink'].item()
        ans_for_display = ans_for_display.replace(link, hyperlink)
    return ans_for_display

def user_input_phase():
    '''
    This is the method for users to choose a topic at the beginning.
    '''
    
    print('Enter 1 for Creating Phase')
    print('Enter 2 for Formalizing Phase')
    print('Enter 3 for Excuting Phase')

    choice = int(input('Enter your choice:'))

    if (choice == 1):
      text_tf = file_io.read_data('creating_phase_corpus.json')
    if (choice == 2):
      text_tf = file_io.read_data('formalizing_phase_corpus.json')
    if (choice == 3):
      text_tf = file_io.read_data('executing_phase_corpus.json')

    else:
        print('Invalid choice')
    return text_tf, choice
    
# initialize the model and the tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
BERT_TOKEN_LIMIT = 512
map_qns_to_num_tokens = {}

# checking the current environment type
print("Current environment: ")
print(platform.uname())

# pick the context from dataset
# index = 14
# original_context = file_io.read_data("official_corpus/executing_phase_corpus.json")["text"][index]

# print the initial context and start conversation
# print("The initial context: " + original_context)
# begin_conversation(original_context, map_qns_to_num_tokens)"""

# use this block of code to be able to feed questions directly instead of typing each time
# data file contains text and list of questions
test_df = file_io.read_data('test_file_to_show.json')

# for i in range(0, test_df.shape[0]):

original_context = test_df["data"][0]["text"]
q_list = test_df["data"][0]["questions"]

test_conversation(original_context, q_list, map_qns_to_num_tokens)
