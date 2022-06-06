import pandas as pd
# import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from time import time
import platform
import re
import file_io
import context_ranker


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

    # checking for a null answer
    if answer.startswith("[CLS]") or answer == " " or answer == "":
        answer = "Unable to find the answer to your question."

    # checking if there is a period at the end of the answer
    last_answer = answer[-1]
    if last_answer != ".":
        answer = answer + "."

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

    confidence_score = 0
    start_prob = 0.0
    end_prob = 0.0

    if arg_start_index == softm_start_index and arg_end_index == softm_end_index:

        start_prob = prob_dist_start[0, softm_start_index]
        end_prob = prob_dist_end[0, softm_end_index]
        confidence_score = 0.7 * start_prob + 0.3 * end_prob

    return confidence_score, start_prob, end_prob


def get_slide_number(context):
    """
    This method can help to get slide number from input context
    :param context: input original context
    """
    if context != "":
        split_content = context.split(r' ')
        slide_number = ' ' + 'Slide' + ' ' + split_content[1]
    else:
        slide_number = -1

    return slide_number


def question_answer(question, cur_context, history, map):
    """
    This is the main code for qa system - uses BERT
    :param question: current question being asked
    :param cur_context: current context used
    :param history: list of questions previously asked
    :param map: current map of question to number of tokens
    :return: current context, current question history, predicted answer
    """

    if len(history) != 0:
        # append past questions to current context
        cur_context = "".join(history) + cur_context

    # finding number of tokens in history and the full context
    all_text_inputs, all_text_input_ids, all_text_text_tokens = get_tokens_helper(cur_context)

    # checking if the length of the context tokens is over 512, since BERT has a restriction
    if len(all_text_text_tokens) >= BERT_TOKEN_LIMIT:

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

        # append truncated history to current context
        cur_context = "".join(history) + cur_context

    # using a helper method to get tokens
    inputs, input_ids, text_tokens = get_tokens_helper(question, cur_context)

    # sending tokens to BERT model
    model_out = model(**inputs)

    # getting starting ending indices for predicted answer
    answer_start = torch.argmax(model_out.start_logits)
    answer_end = torch.argmax(model_out.end_logits)

    # using a helper method to check for the answer
    answer = get_answer_helper(answer_start, answer_end, input_ids, text_tokens)

    # check the confidence of the answer, returns a boolean based on threshold
    confidence_components = get_confidence(model_out.start_logits, model_out.end_logits)

    """if not confident:
        answer = ""
    else:
        """


    # strings of history and the full context
    cur_history_component = question + " "
    history.append(cur_history_component)

    # creating a map and having it ready with questions mapped to number of tokens
    map = map_question_to_num_tokens(cur_history_component, map)

    # print(f"Current history|context: {history} | {cur_context}")
    return history, answer, map, confidence_components


def test_conversation(phase_df, question_list, map):
    """
    Can use this method to directly pass in a list of questions instead of typing it in
    :param phase_df:
    :param question_list: list of questions you want to ask
    :param map current map of question to number of tokens
    :return: nothing
    """
    history = []
    logging_list = []
    for i, question in enumerate(question_list):
        print("\n------------------------------------------------------")
        print(f"{question}")
        list_candidates = context_ranker.get_top_k_file(question, phase_df, 2)

        for j in range(0, 2):
            cur_context = list_candidates[j]
            time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, cur_context, history, map)
            # print("Time taken: " + str(time_taken_to_answer))

            confidence_components = qa_returned_elements[3]
            current_answer = qa_returned_elements[1]
            cscore = confidence_components[0]
            start_prob = confidence_components[1]
            end_prob = confidence_components[2]

            print("------------------------------------------------------")
            print("\nPredicted answer:\n{}".format(current_answer.capitalize()))
            print(f"Weighted confidence: {cscore}")
            print("Confidence start score: " + str(start_prob) + ", end score: " + str(end_prob))
            print("For more information, please go to" + get_slide_number(cur_context))

            # logging
            logging_list.append([question, ''.join(history), current_answer, str(time_taken_to_answer)])
            logging_df = pd.DataFrame(logging_list,
                                      columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
            if i == 0:
                file_io.write_data(logging_df, first_time=True)
            else:
                file_io.write_data(logging_df)
            logging_list = []
            history = []

        history = qa_returned_elements[0]
        map = qa_returned_elements[2]


def begin_conversation(map, phase_df):
    """
    This method output instructions to begin the conversation
    :param phase_df:
    :param map current map of question to number of tokens
    :return: nothing
    """
    history = []
    question = input("\nPlease enter your question: \n")

    while True:
        # checking if the question is end with a question mark
        last_character = question[-1]
        if last_character != "?":
            question = question + "?"

        cur_context = context_ranker.get_top_k_file(question, phase_df)
        time_taken_to_answer, qa_returned_elements = get_time(question_answer, question, cur_context, history, map)
        print("Time taken: " + str(time_taken_to_answer))
        history = qa_returned_elements[0]
        current_answer = qa_returned_elements[1]
        map = qa_returned_elements[2]
        confidence_components = qa_returned_elements[3]

        cscore = confidence_components[0]
        start_prob = confidence_components[1]
        end_prob = confidence_components[2]

        print("\nPredicted answer:\n{}".format(current_answer.capitalize()))
        print(f"Weighted confidence: {cscore}")
        print("Confidence start score: " + str(start_prob) + ", end score: " + str(end_prob))
        print("For more information, please go to" + get_slide_number(cur_context))

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


def insert_links(answer):
    ans_for_display = answer

    df = pd.read_csv('map_of_hyperlinks.csv')

    list = re.findall("link\d\d", ans_for_display)

    for link in list:
        hyperlink = df.loc[df['Link'] == link]['Hyperlink'].item()
        ans_for_display = ans_for_display.replace(link, hyperlink)
    return ans_for_display


def user_input_phase():
    """
    This is the method for users to choose a topic at the beginning.
    """
    choices_df = file_io.read_data('initial_choices.json')
    print(f"Enter 1 for {choices_df['choice1'][0]}, 2 for {choices_df['choice2'][0]}, "
          f"3 for {choices_df['choice3'][0]}")

    # print("Enter 1 for Creating Phase, 2 for Formalizing Phase, 3 for Executing Phase")
    choice = int(input('Enter your choice: '))
    flag = True
    while flag:
        try:
            ch_string = "choice" + str(choice)
            text_df = file_io.read_data(choices_df[ch_string][1])
            flag = False
        except KeyError:
            print('Invalid choice, try again')
            choice = int(input('Enter your choice:'))
            flag = True
    return text_df


# initialize the model and the tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
BERT_TOKEN_LIMIT = 512
map_qns_to_num_tokens = {}

# checking the current environment type
print("Current environment: ")
print(platform.uname())

current_phase_df = user_input_phase()
# begin_conversation(map_qns_to_num_tokens, current_phase_df)

test_df = file_io.read_data('ranker_and_qa_list.json')

q_list = test_df["data"][2]["questions"]
test_conversation(current_phase_df, q_list, map_qns_to_num_tokens)