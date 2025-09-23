import data
import re
import string

number_string_textual = set(data.NUM_TO_TEXT.values())
number_string_numeric = set(map(str, range(100)))

def split_by_all_punctuations(text):
    pattern = f"[\\s{re.escape(string.punctuation)}]+"  # split on whitespace or any punctuation
    words = [t for t in re.split(pattern, text) if t]  # remove empty strings

    return words

def text_has_single_number(generated_text):
    generated_text = generated_text.lower()
    generated_words = set(split_by_all_punctuations(generated_text))
    numbers_in_text = generated_words & number_string_textual | generated_words & number_string_numeric
    return len(numbers_in_text) == 1

def text_has_answer(generated_text, answer_numeric, answer_textual):
    generated_text = generated_text.lower()
    generated_words = set(split_by_all_punctuations(generated_text))
    return str(answer_numeric) in generated_words or answer_textual in generated_words
