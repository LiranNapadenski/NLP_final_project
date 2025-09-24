import data
import re
import string

TEXTUAL_NUMBER_STRINGS = set(data.NUM_TO_TEXT.values())
NUMERIC_NUMBER_STRINGS = set(map(str, range(100)))


class RepetitionSyndromes:
    BOTH = "both"
    NUM1 = "num1"
    NUM2 = "num2"
    NONE = "none"


def split_by_all_punctuations(text):
    pattern = f"[\\s{re.escape(string.punctuation)}]+"  # split on whitespace or any punctuation
    words = [t for t in re.split(pattern, text) if t]  # remove empty strings

    return words


def text_has_single_number(generated_text):
    return len(numbers_in_text(generated_text)) == 1


def text_has_answer(generated_text, answer_numeric):
    return answer_numeric in numbers_in_text(generated_text)


def numbers_in_text(generated_text):
    generated_text = generated_text.lower()
    generated_words = set(split_by_all_punctuations(generated_text))

    numeric_numbers_in_text = generated_words & NUMERIC_NUMBER_STRINGS
    textual_numbers_in_text = generated_words & TEXTUAL_NUMBER_STRINGS

    nums_in_text = numeric_numbers_in_text | {str(data.TEXT_TO_NUM[text]) for text in textual_numbers_in_text}

    return {int(n) for n in nums_in_text}


def first_number_in_text(generated_text):
    generated_text = generated_text.lower()
    generated_words = set(split_by_all_punctuations(generated_text))

    num = None

    for word in generated_words:
        if word in NUMERIC_NUMBER_STRINGS:
            num = int(word)
            break
        elif word in TEXTUAL_NUMBER_STRINGS:
            num = int(data.TEXT_TO_NUM[word])
            break

    return num


def repetition_syndrome(generated_text_new_tokens_only, prompt):
    first_num = first_number_in_text(generated_text_new_tokens_only)

    if first_num == prompt.num1:
        return RepetitionSyndromes.NUM1
    elif first_num == prompt.num2:
        return RepetitionSyndromes.NUM2
    else:
        return RepetitionSyndromes.NONE


def right_direction(generated_number, prompt):
    if prompt.operator == data.PromptOperators.PLUS:
        return generated_number > prompt.num1
    elif prompt.operator == data.PromptOperators.MINUS:
        return generated_number < prompt.num1

    return None


class EvaluationMetrics:
    def __init__(self, generated_tokens, prompt):
        self.generated_numbers = numbers_in_text(generated_tokens)
        self.first_generated_number = first_number_in_text(generated_tokens)
        self.has_single_number = text_has_single_number(generated_tokens)
        self.has_answer = text_has_answer(generated_tokens, prompt.answer)
        self.repetition_syndrome = repetition_syndrome(generated_tokens, prompt)

        self.diff_from_answer = None
        self.right_direction = None

        if self.first_generated_number is not None:
            self.diff_from_answer = self.first_generated_number - prompt.answer
            self.right_direction = right_direction(self.first_generated_number, prompt)
