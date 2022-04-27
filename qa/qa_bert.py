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


def question_answer(question, context, question_history):
    """
    This is the main code for qa system - uses BERT
    :param question: current question being asked
    :param context: current context used
    :param question_history: list of questions previously asked
    :return: current context and the predicted answer
    """
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    model_out = model(**inputs)

    # reconstructing the answer
    answer_start = torch.argmax(model_out.start_logits)
    answer_end = torch.argmax(model_out.end_logits)

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    if answer_end >= answer_start:
        answer = text_tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if text_tokens[i][0:2] == "##":
                answer += text_tokens[i][2:]
            else:
                answer += " " + text_tokens[i]

    if answer.startswith("[CLS]") or answer == " ":
        answer = "Unable to find the answer to your question."

    print("\nPredicted answer:\n{}".format(answer.capitalize()))

    
    question_history = ''.join(question_history)
    all_text = ''.join(question_history) + context
    
    
    all_text_inputs = tokenizer.encode_plus(all_text, add_special_tokens=True, return_tensors="pt")
    all_text_input_ids = all_text_inputs["input_ids"].tolist()[0]

    all_text_text_tokens = tokenizer.convert_ids_to_tokens(all_text_input_ids)

    question_history_inputs = tokenizer.encode_plus(question_history, add_special_tokens=True, return_tensors="pt")
    question_history_input_ids = question_history_inputs["input_ids"].tolist()[0]

    question_history_text_tokens = tokenizer.convert_ids_to_tokens(question_history_input_ids)
    
    # print(len(all_text_text_tokens))
    if len(all_text_text_tokens) > 70:
        tokens_over_limit = len(all_text_text_tokens) - 70
        #print(tokens_over_limit)
        tokens_to_remove = len(question_history_text_tokens) - tokens_over_limit - 1
        #print(tokens_to_remove)
        question_history = question_history_text_tokens[1:tokens_to_remove]
        #print(question_history)
        all_text = ' '.join(question_history) + context
        print('All text before tokenization:', all_text)

        '''
        # This is to see how many tokens will be after the cut 
         
        all_text_inputs = tokenizer.encode_plus(all_text, add_special_tokens=True, return_tensors="pt")
        all_text_input_ids = all_text_inputs["input_ids"].tolist()[0]

        all_text_text_tokens = tokenizer.convert_ids_to_tokens(all_text_input_ids)

        print('All text tokens after cut:', all_text_text_tokens)
        print(len(all_text_text_tokens))
        '''
    # print("Current context: " + all_text)
    return context, answer


def test_conversation(initial_context, question_list):
    """
    Can use this method to directly pass in a list of questions instead of typing it in
    :param initial_context: the initial piece of context
    :param question_list: list of questions you want to ask
    :return: nothing
    """
    question_history = []
    logging_list =[]
    text = initial_context

    for i, question in enumerate(question_list):
        """if question + " " not in question_history:
            question_history.append(question + " ")"""
        question_history.append(question + " ")
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, question_history)
        current_answer = qa_returned_elements[1]

        # logging
        logging_list.append([question, ''.join(question_history), current_answer, str(time_taken_to_answer)])
        logging_df = pd.DataFrame(logging_list, columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
        if i == 0:
            file_io.write_data(logging_df, first_time=True)
        else:
            file_io.write_data(logging_df)
        logging_list = []


def begin_conversation(initial_context):
    """
    This method output instructions to begin the conversation
    :param initial_context: the inital piece of text
    :return: nothing
    """
    question_history = []
    # logging_list =[]

    question = input("\nPlease enter your question: \n")
    question_history.append(question + " ")
    text = initial_context

    while True:
        # checking the time taken by each call
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, text, question_history)
        print("Time taken: " + str(time_taken_to_answer))
        # current_context = qa_returned_elements[0]
        current_answer = qa_returned_elements[1]

        """logging
        logging_list.append([question, ''.join(question_history), current_answer, str(time_taken_to_answer)])
        logging_df = pd.DataFrame(logging_list, columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
        file_io.write_data(loggisng_df)
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

# pick the context from dataset
index = 0
initial_text = file_io.read_data('sample_dataset.json')["text"][index]

# checking the current environment type
print("Current environment: ")
print(platform.uname())

# print the initial context and start conversation
print("The initial context: " + initial_text)
# begin_conversation(initial_text)


# use this block of code to be able to feed questions directly instead of typing each time
# data file contains text and list of questions
test_df = file_io.read_data('test_file_to_show.json')
for i in range(0, test_df.shape[0]):
    initial_context = test_df["data"][i]["text"]
    q_list = test_df["data"][i]["questions"]
    test_conversation(initial_context, q_list)
