import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import platform
import re
import file_io
import context_ranker
import check_spelling_grammar
import logging_conversation as logger
import utils as utils


def question_answer(question, cur_context, history, map, tokenizer, model):
    """
    This is the main code for qa system - uses BERT
    :param model:
    :param tokenizer:
    :param question: current question being asked
    :param cur_context: current context used
    :param history: list of questions previously asked
    :param map: current map of question to number of tokens
    :return: current context, current question history, predicted answer
    """
    BERT_TOKEN_LIMIT = 512

    if len(history) != 0:
        # append past questions to current context
        cur_context = "".join(history) + cur_context

    # finding number of tokens in history and the full context
    all_text_inputs, all_text_input_ids, all_text_text_tokens = utils.get_tokens_helper(cur_context, tokenizer)

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
    inputs, input_ids, text_tokens = utils.get_tokens_helper(question, cur_context, tokenizer)

    # sending tokens to BERT model
    model_out = model(**inputs)

    # getting starting ending indices for predicted answer
    answer_start = torch.argmax(model_out.start_logits)
    answer_end = torch.argmax(model_out.end_logits)

    # using a helper method to check for the answer
    answer = utils.get_answer_helper(answer_start, answer_end, input_ids, text_tokens, tokenizer)

    # check the confidence of the answer, returns a boolean based on threshold
    confidence_components = utils.get_confidence(model_out.start_logits, model_out.end_logits)

    # strings of history and the full context
    cur_history_component = question + " "
    history.append(cur_history_component)

    # creating a map and having it ready with questions mapped to number of tokens
    map = utils.map_question_to_num_tokens(cur_history_component, map, tokenizer)

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

        question = question.lower()
        question = check_spelling_grammar.check_spelling(question)
        question = check_spelling_grammar.check_grammar(question)

        num_potential_candidates = 3
        list_candidates = context_ranker.get_top_k_file(question, phase_df, num_potential_candidates)

        confidence_candidates = []
        answer_candidates = []
        map_candidates = []

        for j in range(0, num_potential_candidates):

            cur_context = list_candidates[j]

            # this is to ensure bert doesn't confuse the slide number in the context
            # with a quantifiable number in a question where we don't talk about
            # a slide number or where you can find something
            if "slide" not in question and "where" not in question:
                cur_context = re.sub('Slide \d*', '', cur_context)

            time_taken_to_answer, qa_returned_elements = utils.get_time(question_answer, question, cur_context, history,
                                                                        map)
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
            print("For more information, please go to" + utils.get_slide_number(cur_context))

            # logging
            if i == 0:  # this needs to change as well
                logging_list = logger.log(logging_list, question, history, current_answer, time_taken_to_answer,
                                          initial=True)
            else:
                logging_list = logger.log(logging_list, question, history, current_answer, time_taken_to_answer)

            if type(cscore) is float:
                confidence_candidates.append(cscore)
            else:
                confidence_candidates.append(cscore.item())
            answer_candidates.append(current_answer)
            map_candidates.append(qa_returned_elements[2])
            history = []

        accepted_answer_index = np.argmax(np.array(confidence_candidates))
        history.append(question)
        map = map_candidates[accepted_answer_index]


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

        question = question.lower()
        question = check_spelling_grammar.check_spelling(question)
        question = check_spelling_grammar.check_grammar(question)

        num_potential_candidates = 3
        list_candidates = context_ranker.get_top_k_file(question, phase_df, num_potential_candidates)

        confidence_candidates = []
        answer_candidates = []
        map_candidates = []

        for j in range(0, num_potential_candidates):

            cur_context = list_candidates[j]

            # this is to ensure bert doesn't confuse the slide number in the context
            # with a quantifiable number in a question where we don't talk about
            # a slide number or where you can find something
            if "slide" not in question and "where" not in question:
                copy_original_context = cur_context
                cur_context = re.sub('Slide \d*', '', cur_context)

            time_taken_to_answer, qa_returned_elements = utils.get_time(question_answer, question, cur_context, history,
                                                                        map)
            # print("Time taken: " + str(time_taken_to_answer))

            confidence_components = qa_returned_elements[3]
            current_answer = qa_returned_elements[1]
            cscore = confidence_components[0]
            start_prob = confidence_components[1]
            end_prob = confidence_components[2]

            if type(cscore) is float:
                confidence_candidates.append(cscore)
            else:
                confidence_candidates.append(cscore.item())
            answer_candidates.append(current_answer)
            map_candidates.append(qa_returned_elements[2])
            history = []

        accepted_answer_index = np.argmax(np.array(confidence_candidates))
        history.append(question)
        map = map_candidates[accepted_answer_index]

        print("------------------------------------------------------")
        print("\nPredicted answer:\n{}".format(answer_candidates[accepted_answer_index].capitalize()))
        print(f"Weighted confidence: {confidence_candidates[accepted_answer_index]}")
        print("For more information, please go to" + utils.get_slide_number(copy_original_context))

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


def user_input_phase():
    """
    This is the method for users to choose a topic at the beginning.
    """
    choices_df = file_io.read_data_json('official_corpus/initial_choices.json')
    print(f"Enter 1 for {choices_df['choice1'][0]}, 2 for {choices_df['choice2'][0]}, "
          f"3 for {choices_df['choice3'][0]}")

    choice = int(input('Enter your choice: '))
    flag = True
    while flag:
        try:
            ch_string = "choice" + str(choice)
            text_df = file_io.read_data_json(choices_df[ch_string][1])
            flag = False
        except KeyError:
            print('Invalid choice, try again')
            choice = int(input('Enter your choice:'))
            flag = True
    return text_df


if __name__ == '__main__':
    # initialize the model and the tokenizer
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    map_qns_to_num_tokens = {}

    # checking the current environment type
    print("Current environment: ")
    print(platform.uname())

    current_phase_df = user_input_phase()
    begin_conversation(map_qns_to_num_tokens, current_phase_df, bert_tokenizer, bert_model)

    # test_df = file_io.read_data('official_corpus/ranker_and_qa_list.json')

    # q_list = test_df["data"][0]["questions"]
    # test_conversation(current_phase_df, q_list, map_qns_to_num_tokens)
