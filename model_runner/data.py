import random
import yaml
import os

NUM_TO_TEXT = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}
NUM_TO_TEXT_MAX = 20

class NumberFormats:
    TEXTUAL = "textual"
    NUMERIC = "numeric"

class DatasetTypes:
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"

class NumberRanges:
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# Number ranges
MAX_NUMBERS = {
    NumberRanges.SMALL: 8,
    NumberRanges.MEDIUM: 20,
    NumberRanges.LARGE: 70,
}

class PromptOperators:
    PLUS = "+"
    MINUS = "-"


def format_num(num, number_format):
    if num <= NUM_TO_TEXT_MAX and number_format == NumberFormats.TEXTUAL:
        return NUM_TO_TEXT[num]
    else:
        return str(num)


class PromptTemplate:
    question_format = "Question: {question_text} \nAnswer:"
    def __init__(self, id, template, addition_verbs, subtraction_verbs, items):
        self.id = id
        self.template = template
        self.addition_verbs = addition_verbs
        self.subtraction_verbs = subtraction_verbs
        self.items = items

    def generate_all_prompts(self, max_number, number_format):
        prompts = []
        if len(self.items) == 0:
            self.items = [""]

        for operation, verbs in [(PromptOperators.PLUS, self.addition_verbs), (PromptOperators.MINUS, self.subtraction_verbs)]:
            for verb in verbs:
                for num1 in range(1, max_number):
                    for item in self.items:
                        item1 = item if num1 != 1 else item[:-1]
                        for num2 in range(1, num1):

                            if num1 > NUM_TO_TEXT_MAX: # If number is too large, there is no choice
                                number_formats = [NumberFormats.NUMERIC]
                            elif number_format is None: # If caller has no preference, do both
                                number_formats = [NumberFormats.TEXTUAL, NumberFormats.NUMERIC]
                            else: # If caller has a preference, do only the preference
                                number_formats = [number_format]

                            for number_format in number_formats:
                                item2 = item if num2 > 1 else item[:-1]
                                num1_str = format_num(num1, number_format)
                                num2_str = format_num(num2, number_format)

                                question_text = self.template.format(num1=num1_str,
                                                                     num2=num2_str,
                                                                     item1=item1,
                                                                     item2=item2,
                                                                     verb=verb)

                                prompt_text = self.question_format.format(question_text=question_text)

                                prompts.append(Prompt(template_id=self.id,
                                                      text=prompt_text,
                                                      operator=operation, num1=num1, num2=num2,
                                                      items=item, verb=verb, number_format=number_format))

        return prompts

    def generate_prompts(self, num_of_prompts, max_number, number_format=None):
        prompts = []

        if len(self.items) == 0:
            self.items = [" "]

        randomize_number_format = True if number_format is None else False

        for operation, verbs in [(PromptOperators.PLUS, self.addition_verbs),
                                 (PromptOperators.MINUS, self.subtraction_verbs)]:
            for verb in verbs:
                for _ in range(num_of_prompts):
                    num1 = random.randint(2, max_number)
                    num2 = random.randint(1, num1-1)

                    if randomize_number_format:
                        if num1 > NUM_TO_TEXT_MAX:
                            number_format = NumberFormats.NUMERIC
                        else:
                            number_format = random.choice([NumberFormats.TEXTUAL, NumberFormats.NUMERIC])

                    item = random.choice(self.items)
                    item1 = item if num1 > 1 else item[:-1]
                    item2 = item

                    num1_str = format_num(num1, number_format)
                    num2_str = format_num(num2, number_format)

                    question_text = self.template.format(num1=num1_str,
                                                         num2=num2_str,
                                                         item1=item1,
                                                         item2=item2,
                                                         verb=verb)

                    prompt_text = self.question_format.format(question_text=question_text)

                    prompts.append(Prompt(template_id=self.id,
                                          text=prompt_text,
                                          operator=operation, num1=num1, num2=num2,
                                          items=item, verb=verb, number_format=number_format))

        return prompts


class Prompt:
    def __init__(self, template_id, text, operator, num1, num2, items, verb, number_format):
        self.template_id = template_id
        self.text = text
        self.num1 = num1
        self.operator = operator
        self.num2 = num2
        self.answer = num1 + num2 if operator == '+' else num1 - num2
        self.answer_str = NUM_TO_TEXT[self.answer] if self.answer <= NUM_TO_TEXT_MAX else ""
        self.number_format = number_format
        self.items = items
        self.verb = verb
        self.id = 0 # To be set by dataset builder


# Core dataset builder
def build_dataset(dataset_type, number_range_key, prompts_per_verb=20, generate_all_prompts=False):
    """
    Generates prompts_per_verb prompts per template per operator (+/-), total of 2 * prompts_per_verb * (number of templates in yml)
    :param dataset_type: "implicit" or "explicit" phrasing of the question
    :param number_range_key: "small" or "medium" or "large", the scale of the numbers in the questions
    :param prompts_per_verb: the number of prompts to generate for each question template
    :generate_all_prompts: whether to generate all possible prompts in each template
    """
    templates_yml_path = "datasets/" + dataset_type + "_dataset.yml"
    script_dir = os.path.dirname(os.path.realpath(__file__)) 
    templates_yml_path = os.path.join(script_dir, templates_yml_path)
    
    with open(templates_yml_path, "r") as f:

        # Load YAML into custom objects
        yml_templates = yaml.safe_load(f)
        templates = [PromptTemplate(**yml_template) for yml_template in yml_templates]

        prompts = []
        max_num = MAX_NUMBERS[number_range_key]

        number_format = NumberFormats.NUMERIC if dataset_type == DatasetTypes.EXPLICIT else None

        if generate_all_prompts:
            for template in templates:
                prompts += template.generate_all_prompts(max_num, number_format)
        else:
            for template in templates:
                prompts += template.generate_prompts(prompts_per_verb, max_num, number_format)

        prompt_id = 0
        for p in prompts:
            p.id = prompt_id
            prompt_id += 1

        return prompts


# Factory function
def dataset_factory(dataset_name: str, prompts_per_verb=20, generate_all_prompts=False):
    """
    Generate different datasets based on dataset_name.
    """

    dataset_type = dataset_name.split("_")[0]  # explicit or implicit
    number_range_key = dataset_name.split("_")[1]  # small medium or large

    if (dataset_type in [DatasetTypes.IMPLICIT, DatasetTypes.EXPLICIT] and
            number_range_key in [NumberRanges.SMALL, NumberRanges.MEDIUM, NumberRanges.LARGE]):

        return build_dataset(dataset_type, number_range_key, prompts_per_verb, generate_all_prompts)

    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")


if __name__ == "__main__":
    # Test
    prompts = dataset_factory("implicit_small", prompts_per_verb=20, generate_all_prompts=False)
    for p in prompts:
        pass
    sett = {'6'}
    print(sett.pop())