import pandas as pd
import numpy as np
import re
import difflib
import file_io as file_io
import check_spelling_grammar as check_spelling_grammar
import context_ranker as context_ranker
import question_answer_bot as qa_bot
import logging_conversation as logger
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import utils as utils
import random


def find_match(text1, text2):
    """
    Uses SequenceMatcher to check for the longest human-like sequence between 2 strings
    and uses ratio to find a similarity score between 0 and 1
    :param text1: text to be compared
    :param text2: text to be compared
    :return: True if the ratio is over 50%
    """
    s = difflib.SequenceMatcher(None, text1, text2)
    score = s.ratio()
    if score > 0.5:
        return score, 1
    else:
        return score, 0


def read_data_and_feed_to_bert(filename_q_bank, filename_corpus, output_file):
    """
    Reads data, feeds into bert, gets predicted answer and compares that to actual answer to get
    a match score
    :param filename_q_bank: filename for question bank
    :param filename_corpus: filename for corpus
    :param output_file: file to write output to
    :return: dataframw with predicted answer and score
    """
    map = {}

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    qna_bank_df = file_io.read_data_csv(filename_q_bank)
    phase_df = file_io.read_data_json(filename_corpus)

    qna_bank_df['Questions'] = qna_bank_df['Questions'].apply(lambda x: x.lower())

    eval_list = []
    question_list = qna_bank_df['Questions'].to_list()
    expected_answers_list = qna_bank_df['Expected Answer'].to_list()
    history = []
    predicted_answers_list = []
    score = []

    for i, question in enumerate(question_list):
        # if i == 16:
        #     print("here is question 17 with an issue")
        if "mou" not in question:
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
            # a 'slide' number or 'where' you can find something
            if "slide" not in question and "where" not in question:
                # copy_original_context = cur_context
                cur_context = re.sub('Slide \d*', '', cur_context)

            time_taken_to_answer, qa_returned_elements = utils.get_time(qa_bot.question_answer, question, cur_context,
                                                                        history,
                                                                        map, tokenizer, model)
            confidence_components = qa_returned_elements[3]
            current_answer = qa_returned_elements[1]
            cscore = confidence_components[0]

            if type(cscore) is float:
                confidence_candidates.append(cscore)
            else:
                confidence_candidates.append(cscore.item())
            answer_candidates.append(current_answer)
            map_candidates.append(qa_returned_elements[2])
            history = []

        accepted_answer_index = np.argmax(np.array(confidence_candidates))
        history = []
        map = map_candidates[accepted_answer_index]

        predicted_answer = answer_candidates[accepted_answer_index]
        predicted_answers_list.append(predicted_answer)

        comp = find_match(expected_answers_list[i], predicted_answer)
        score.append(comp[0])
        isCorrect = comp[1]
        eval_list.append(isCorrect)

        print(f"Question {i+1} done")

    final_df = pd.DataFrame({'Question': question_list, 'Expected Answer': expected_answers_list,
                             'Predicted Answer': predicted_answers_list, 'Score': score, 'isCorrect': eval_list})

    file_io.write_data(final_df, output_file, first_time=True)

    return final_df


def get_benchmark_set(filename_q_bank, filename_corpus, output_file):

    qna_bank_df = file_io.read_data_csv(filename_q_bank)
    phase_df = file_io.read_data_json(filename_corpus)

    eval_list = []
    question_list = qna_bank_df['Questions'].to_list()
    expected_answers_list = qna_bank_df['Expected Answer'].to_list()
    random_answers_list = []
    score = []

    for i, question in enumerate(question_list):

        # randomly pick a context
        randomly_selected_context = ''.join(phase_df.sample(n=1, random_state=1)['text'].tolist())

        # randomly choose a span of text as answer
        # as checked, min answer length is 3, max length is 269, avg length is 64
        mu = 64
        sigma = 5 # choosing this value
        random_start_index = random.randint(0, len(randomly_selected_context))
        random_offset = int(np.random.normal(mu, sigma, 1))
        random_end_index = random_start_index + random_offset

        random_answer = randomly_selected_context[random_start_index:random_end_index]

        comp = find_match(expected_answers_list[i], random_answer)
        score.append(comp[0])
        isCorrect = comp[1]
        eval_list.append(isCorrect)
        random_answers_list.append(random_answer)

        print(f"Question {i+1} done!")

    final_df = pd.DataFrame({'Question': question_list, 'Expected Answer': expected_answers_list,
                             'Random Answer': random_answers_list, 'Score': score, 'isCorrect': eval_list})

    file_io.write_data(final_df, output_file, first_time=True)

    return final_df

def min_max_avg_len_answer(df_1, df_2, df_3):
    """

    """
    df = pd.concat([df_1, df_2, df_3])
    predicted_answers = df['Predicted Answer'].tolist()
    min_len = len(min(predicted_answers, key=len))
    max_len = len(max(predicted_answers, key=len))
    avg_len = int(sum(map(len, predicted_answers))/float(len(predicted_answers)))

    return min_len, max_len, avg_len

def evaluate(df):
    """
    Check how many BERT got correct
    :param df: passing dataframe with actual, expected, score
    :return: evaluation score
    """
    num_correct = len(df.query('isCorrect == 1'))
    num_total = len(df)
    eval_score = num_correct / num_total
    return eval_score


if __name__ == '__main__':
    corpus_df = file_io.read_data_json("official_corpus/initial_choices.json")
    choice = "choice3"
    filename_to_corpus = corpus_df[choice][1]
    filename_to_qbank = corpus_df[choice][2]

    filename_to_write_to = "3rd_executing_benchmark_eval.csv"

    benchmark_answers_df = get_benchmark_set(filename_to_qbank, filename_to_corpus, filename_to_write_to)
    score = evaluate(benchmark_answers_df)
    print(f"Benchmark accuracy on {corpus_df[choice][0]} question bank is: {score} ")

    # total_time_taken, answer_df = utils.get_time(read_data_and_feed_to_bert, filename_to_qbank, filename_to_corpus, filename_to_write_to)
    #
    # score = evaluate(answer_df)
    # print(f"Accuracy on {corpus_df['choice3'][0]} question bank is: {score} ")
    # print(f"Total time taken to evaluate: {total_time_taken} ")

    # df1 = file_io.read_data_csv(corpus_df['choice1'][3])
    # df2 = file_io.read_data_csv(corpus_df['choice2'][3])
    # df3 = file_io.read_data_csv(corpus_df['choice3'][3])
    #
    # ret_comp = min_max_avg_len_answer(df1, df2, df3)
    # min_len_ans = ret_comp[0] # 3
    # max_len_ans = ret_comp[1] # 269
    # avg_len_ans = ret_comp[2] # 64

