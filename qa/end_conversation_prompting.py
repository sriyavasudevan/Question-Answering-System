import random
import textwrap
import file_io


def yes_or_no():
    """

    :return:
    """
    print('Are you interested in other learning materials?')
    answer_choice = input('Please enter Y or N: ')
    print('\n')
    if answer_choice == 'Y' or answer_choice == 'y':
        continue_q = 1
    else:
        print('Bye!')
        continue_q = 0
    return continue_q


def prompting_further_info(choice, continue_q):
    """

    :param choice:
    :param continue_q:
    :return:
    """
    df = file_io.read_data_json('official_corpus/initial_choices.json')

    option1_cs = df['resources'][0]
    option2_cl = df['resources'][1]
    option3_npi = df['resources'][2]
    phs_nm = [df['choice1'][0],
              df['choice2'][0],
              df['choice3'][0]]

    if continue_q == 1:
        print('Please choose from the following resources: ')
        print(f'Option 1: {option1_cs}')
        print(f'Option 2: {option2_cl}')
        if choice != 3:
            print(f'Option 3: {option3_npi}')

        print('\n')
        end_choice = int(input('Please enter your choice: '))
        print('\n')
        if end_choice == 1:
            ending = 'You are going to learn more about the ' + option1_cs + ' of ' + phs_nm[choice - 1]
        elif end_choice == 2:
            ending = 'You are going to learn more about the ' + option2_cl + ' of ' + phs_nm[choice - 1]
        elif end_choice == 3:
            ending = 'You are going to learn more about the ' + option3_npi + phs_nm[choice]
        else:
            ending = 'null'
        print(ending)
    else:
        ending = 'null'
        end_choice = 4
    print('\n')
    return ending, end_choice


def prompting_answer(choice, end_choice):
    """

    :param choice:
    :param end_choice:
    :return:
    """
    df = file_io.read_data_json('official_corpus/initial_choices.json')

    phs_nm = [df['choice1'][0],
              df['choice2'][0],
              df['choice3'][0]]

    chk_path = df['resources'][3]
    ch1_c_path = df['choice1'][4]
    ch2_c_path = df['choice2'][4]
    ch3_c_path = df['choice3'][4]
    npi1 = df['choice1'][5]
    npi2 = df['choice2'][5]

    chk_l = file_io.read_data_json(chk_path)
    ch1_c = file_io.read_data_json(ch1_c_path)
    ch2_c = file_io.read_data_json(ch2_c_path)
    ch3_c = file_io.read_data_json(ch3_c_path)
    cs = {phs_nm[0]: ch1_c, phs_nm[1]: ch2_c, phs_nm[2]: ch3_c}

    npi_l = [npi1, npi2]

    if end_choice == 1:
        current = cs[phs_nm[choice - 1]]
        random_state = random.randint(0, len(current) - 1)
        print(textwrap.fill(current['text'][random_state], 100))
    elif end_choice == 2:
        current = chk_l['text'][choice - 1]
        print(textwrap.fill(current, 100))
    elif end_choice == 3:
        current = npi_l[choice - 1]
        print(textwrap.fill(current, 100))
    else:
        print('Thanks for your inquiry!')
    print('\n')
    print('Have a nice day!')


# give a try
# choice1 = 3
# c_q = yes_or_no()
# ending, end_choice = prompting_further_info(choice1, c_q)
# prompting_answer(choice1, end_choice)
