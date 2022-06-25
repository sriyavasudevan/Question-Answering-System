import pandas as pd
import file_io as file_io


def log(logging_list, question, history, current_answer, time_taken_to_answer, initial=False):
    logging_list.append([question, ''.join(history), current_answer, str(time_taken_to_answer)])
    logging_df = pd.DataFrame(logging_list,
                              columns=['Current Question', 'Question History', 'Predicted Answer', 'Time taken'])
    if initial:
        file_io.write_data(logging_df, first_time=True)
    else:
        file_io.write_data(logging_df)
    logging_list = []
    return logging_list