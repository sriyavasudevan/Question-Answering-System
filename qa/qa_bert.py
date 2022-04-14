import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


def read_data(filename):
    """
    This method is for reading in the data
    """
    sample_text = pd.read_json(filename)
    return sample_text


def question_answer(question, context, question_history):
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

    context = ''.join(question_history) + context
    # print("----")
    # print("The current context is after question: " + context)


def begin_conversation(inital_context):
    question_history = []
    question = input("\nPlease enter your question: \n")
    question_history.append(question)
    text = inital_context

    while True:
        question_answer(question, text, question_history)

        flag = True
        flag_N = False

        while flag:
            response = input("\nDo you want to ask another question based on this text (Y/N)? ")
            if response[0] == "Y":
                question = input("\nPlease enter your question: \n")
                question_history.append(question)

                flag = False
            elif response[0] == "N":
                print("\nBye!")
                question_history = []
                flag = False
                flag_N = True

        if flag_N:
            break


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

index = 0
inital_text = read_data('sample_dataset.json')["text"][index]
print("The initial context: " + inital_text)

begin_conversation(inital_text)

"""
This is a test.
"""