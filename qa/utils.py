import pandas as pd
import torch
from time import time
import re
import sys

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

    if len(args) != 2 and len(args) != 3:
        print("Issue: Error in tokenization")
    else:
        if len(args) == 3:
            question = args[0]
            context = args[1]
            tokenizer = args[2]
            inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        elif len(args) == 2:
            to_be_tokenized = args[0]
            tokenizer = args[1]
            inputs = tokenizer.encode_plus(to_be_tokenized, add_special_tokens=True, return_tensors="pt")

        input_ids = inputs["input_ids"].tolist()[0]
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return inputs, input_ids, text_tokens


def get_answer_helper(answer_start, answer_end, input_ids, text_tokens, tokenizer):
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
        answer = "Unable to find the answer to your question. Please rephrase your question."

    # checking if there is a period at the end of the answer
    last_answer = answer[-1]
    if last_answer != ".":
        answer = answer + "."

    return answer


def map_question_to_num_tokens(question, map, tokenizer):
    """
    a helper method to map the question to number of tokens in that question
    :param question: the question to be tokenized
    :param map: current map of question to number of tokens
    :return: map of question to number of tokens
    """

    # need to get the number of tokens in each question
    question_inputs, question_input_ids, question_text_tokens = get_tokens_helper(question, tokenizer)

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

    confidence_score = 0.0
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


def insert_links(answer):
    """
    Swaps out "link##" for the actual hyperlink associated with certain words
    in the predicted answer
    :param answer: (String) answer to question before replacing link
    :return: (String) answer to question after replacing link
    """
    ans_for_display = answer
    df = pd.read_csv('official_corpus/map_of_hyperlinks.csv')
    link_list = re.findall("link\d\d", ans_for_display)

    for link in link_list:
        hyperlink = df.loc[df['Link'] == link]['Hyperlink'].item()
        ans_for_display = ans_for_display.replace(link, hyperlink)
    return ans_for_display
