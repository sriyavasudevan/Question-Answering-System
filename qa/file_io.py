import pandas as pd


def read_data(filename):
    """
    Helper method to read in json file and convert into dataframe
    :param filename: name of file
    :return: dataframe
    """
    sample_text = pd.read_json(filename)
    return sample_text


def write_data(conversation_df, first_time=False):
    """
    Helper method to write dataframe into a csv
    :param conversation_df: dataframe to be converted
    :param first_time: check if its the first time this method is called
    :return: nothing
    """
    filename = 'log/conversation_log.csv'
    """if os.path.isfile(filename):
        os.remove(filename)"""
    if first_time:
        conversation_df.to_csv(filename, mode='a')
    else:
        conversation_df.to_csv(filename, mode='a', header=False)

