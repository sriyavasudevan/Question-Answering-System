import pkg_resources
from symspellpy import SymSpell, Verbosity
from happytransformer import HappyTextToText, TTSettings


def check_spelling(sequence):
    """
    Uses SymSpell to check & correct words from given sentence against a benchmark dictionary.
    This is based on Damerau-Levenshtein distance.
    :param sequence: (String) the sequence you want spell checked
    :return: (String) modified sequence with correct spelling
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    # lookup suggestions for single-word input strings

    modified_seq_list = []
    word_list = sequence.split(" ")
    for word in word_list:
        # max edit distance per lookup
        # (max_edit_distance_lookup <= max_dictionary_edit_distance)
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)

        """for suggestion in suggestions:
            print(suggestions)"""

        if len(suggestions) == 0:
            modified_seq_list.append(word)
        else:
            modified_seq_list.append(suggestions[0].term)

    typo_corrected_seq = ' '.join(modified_seq_list)
    return typo_corrected_seq


def check_grammar(sentence):
    """
    Checks the structural grammar of a given sentence using t5-base-grammar-correction
    in happytransformers in huggingface
    :param sentence: (String) the sentence you want to check grammar for
    :return: (String) Grammatically correct sentence
    """
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)

    # Add the prefix "grammar: " before each input
    result = happy_tt.generate_text("grammar: " + sentence, args=args)

    return result.text