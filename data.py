import random
import pandas as pd
import re


#add more databasess , make this one better , and add diffrent kinds
#add all of them to the factory
random.seed(42)

# Templates for arithmetic tasks
TEMPLATES = {
    "addition_explicit": [
        "What is {a} + {b}?", "Calculate {a} plus {b}.", "Add {a} and {b} together."
    ],
    "subtraction_explicit": [
        "What is {a} - {b}?", "Subtract {b} from {a}.", "Calculate the difference between {a} and {b}."
    ],
    "multiplication_explicit": [
        "What is {a} * {b}?", "Multiply {a} by {b}.", "What is the product of {a} and {b}?"
    ],
    "division_explicit": [
        "What is {a} / {b}?", "Divide {a} by {b}.", "How many times does {b} fit into {a}?"
    ],
    "addition_implicit": [
        "I had {a} apples, then I got {b} more. How many apples now?",
        "There were {a} kids at the park, {b} more arrived. How many kids in total?"
    ],
    "subtraction_implicit": [
        "I had {a} candies, ate {b}. How many left?",
        "There were {a} players, {b} left. How many remain?"
    ]
}

# Number ranges
RANGES = {
    "small": list(range(0, 21)),
    "medium": list(range(0, 101)),
    "large": list(range(0, 1001)),
    "negatives": list(range(-50, 51))
}

def compute_answer(op, a, b):
    if op.startswith("addition"):
        return a + b
    if op.startswith("subtraction"):
        return a - b
    if op.startswith("multiplication"):
        return a * b
    if op.startswith("division"):
        if b == 0:
            return None
        return round(a / b, 4)
    raise ValueError(f"Unknown operation {op}")

# Core dataset builder
def build_dataset(templates_keys, number_range_key, n_per_combo=20):
    """
    Build a dataset from given template keys and number range.
    """
    rows = []
    idx = 0
    for op_key in templates_keys:
        templates = TEMPLATES[op_key]
        base_op = re.split("_", op_key)[0]
        template_type = "explicit" if "explicit" in op_key else "implicit"
        pool = RANGES[number_range_key]
        for _ in range(n_per_combo):
            a, b = random.choice(pool), random.choice(pool)
            if base_op == "division" and b == 0:
                b = random.choice([x for x in pool if x != 0])
            text = random.choice(templates).format(a=a, b=b)
            ans = compute_answer(op_key, a, b)
            if ans is None:
                continue
            rows.append({
                "id": idx,
                "template_type": template_type,
                "operation": base_op,
                "difficulty": number_range_key,
                "a": a,
                "b": b,
                "text": text,
                "answer": ans,
                "answer_str": str(ans)
            })
            idx += 1
    return pd.DataFrame(rows)

# Factory function
def dataset_factory(dataset_name: str, n_per_combo=20):
    """
    Generate different datasets based on dataset_name.
    """
    if dataset_name == "arithmetic_explicit_small":
        templates_keys = ["addition_explicit", "subtraction_explicit", "multiplication_explicit", "division_explicit"]
        number_range_key = "small"
    elif dataset_name == "arithmetic_implicit_medium":
        templates_keys = ["addition_implicit", "subtraction_implicit"]
        number_range_key = "medium"
    elif dataset_name == "arithmetic_large_all_ops":
        templates_keys = ["addition_explicit", "subtraction_explicit", "multiplication_explicit", "division_explicit"]
        number_range_key = "large"
    elif dataset_name == "arithmetic_negatives":
        templates_keys = ["addition_explicit", "subtraction_explicit"]
        number_range_key = "negatives"
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")
    
    return build_dataset(templates_keys, number_range_key, n_per_combo)
