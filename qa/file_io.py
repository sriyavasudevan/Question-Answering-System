import pandas as pd


def read_data_json(filename):
    """
    Helper method to read in json file and convert into dataframe
    :param filename: name of file
    :return: dataframe
    """
    sample_text = pd.read_json(filename, orient=str)
    return sample_text


def read_data_csv(filename):
    """
    Helper method to read in csv file and convert into dataframe
    :param filename: name of file
    :return: dataframe
    """
    sample_text = pd.read_csv(filename)
    return sample_text


def write_data(df, filename, first_time=False):
    """
    Helper method to write dataframe into a csv
    :param filename:
    :param df: dataframe to be converted
    :param first_time: check if its the first time this method is called
    :return: nothing
    """

    if first_time:
        df.to_csv(filename)
    else:
        df.to_csv(filename, header=False)
