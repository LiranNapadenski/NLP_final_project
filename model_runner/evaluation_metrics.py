import data

num_strings = set(data.NUM_TO_TEXT.values())
nums = set(map(str, range(100)))

def text_has_single_number(generated_text):
    generated_text = generated_text.lower()
    generated_words = set(generated_text.split())
    numbers_in_text = generated_words & num_strings | generated_words & nums
    return len(numbers_in_text) == 1

def text_has_answer(generated_text, answer_numeric, answer_textual):
    generated_text = generated_text.lower()
    generated_words = generated_text.split()
    return answer_numeric in generated_words or answer_textual in generated_words
