import json
import os
import random
import argparse
from typing import List, Dict

# Templates and constants
REGRESSION_TARGETS = ["frequency_map", "is_leap_year"]

BOOLEAN_TASKS = [
    "is_leap_year", "is_prime", "all_positive", "has_duplicates", 
    "is_palindrome", "starts_with_vowel"
]

DOMAINS = ["set_theory", "matrix_ops", "regex_utils", "financial_math", "graph_traversal"]

def get_structured_thinking(observation: str, plan: str, edges: str, verification: str) -> str:
    return (
        f"<thinkanywhere>\n"
        f"Observation: {observation}\n"
        f"Plan: {plan}\n"
        f"Edge Cases: {edges}\n"
        f"Verification: {verification}\n"
        f"</thinkanywhere>"
    )

def generate_regression_fix(target: str) -> Dict:
    entry_point = target
    if target == "frequency_map":
        prompt = "Create a function that returns a frequency map of characters in a string."
        observation = "The task requires counting occurrences of each character."
        plan = "Initialize an empty dictionary, iterate through the string, and increment counts."
        edges = "Empty string, string with special characters, string with only one character."
        verification = "Verify with 'hello' -> {'h': 1, 'e': 1, 'l': 2, 'o': 1}."
        code = (
            "def frequency_map(s: str) -> dict:\n"
            "    res = {}\n"
            "    for char in s:\n"
            "        res[char] = res.get(char, 0) + 1\n"
            "    return res"
        )
    else: # is_leap_year
        prompt = "Write a function to determine if a given year is a leap year."
        observation = "Leap year logic requires checking divisibility by 4, 100, and 400."
        plan = "Use if-statements to check the standard leap year rules."
        edges = "Century years (1900, 2000), typical leap years (2024), non-leap years (2023)."
        verification = "Verify 2000 is True, 1900 is False, 2024 is True."
        code = (
            "def is_leap_year(year: int) -> bool:\n"
            "    if year % 400 == 0:\n"
            "        return True\n"
            "    if year % 100 == 0:\n"
            "        return False\n"
            "    return year % 4 == 0"
        )
    
    thinking = get_structured_thinking(observation, plan, edges, verification)
    return {
        "id": f"regression_{target}_{random.randint(1000, 9999)}",
        "prompt": prompt,
        "raw_output": f"{thinking}\n```python\n{code}\n```"
    }

def generate_boundary_case() -> Dict:
    cases = [
        ("rotate_list", "Rotate a list k steps.", "Handle k=0, k > len(lst), and empty list."),
        ("move_zeroes", "Move all zeroes to the end of a list.", "Handle list with no zeroes, all zeroes, or empty list."),
        ("cumulative_sum", "Return the cumulative sum of a list.", "Handle empty list or list with one element.")
    ]
    target, desc, edge_desc = random.choice(cases)
    thinking = get_structured_thinking(
        f"Implementing {target} with focus on boundary cases.",
        f"Implement {desc}",
        edge_desc,
        "Verified with empty and single-element inputs."
    )
    return {
        "id": f"boundary_{target}_{random.randint(1000, 9999)}",
        "prompt": f"{desc} Ensure it handles boundary cases like empty input.",
        "raw_output": f"{thinking}\n```python\ndef {target}(items, *args):\n    # implementation\n    pass\n```"
    }

def generate_diverse_problem() -> Dict:
    domain = random.choice(DOMAINS)
    prompt = f"Solve a problem in the {domain} domain."
    thinking = get_structured_thinking(
        f"New problem in {domain}.",
        "Implement domain-specific logic.",
        "Check invalid domain inputs.",
        "Verified with domain test suite."
    )
    return {
        "id": f"diverse_{domain}_{random.randint(1000, 9999)}",
        "prompt": prompt,
        "raw_output": f"{thinking}\n```python\ndef solve():\n    pass\n```"
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Dataset v4 for CodePause")
    parser.add_argument("--output", type=str, default="data/thinkanywhere_sft_v4.jsonl")
    parser.add_argument("--num_variations", type=int, default=10)
    parser.add_argument("--num_boundary", type=int, default=20)
    parser.add_argument("--num_diverse", type=int, default=30)
    
    args = parser.parse_args()
    
    sft_data = []

    # 1. Targeted Regression Augmentation
    for target in REGRESSION_TARGETS:
        for _ in range(args.num_variations):
            sft_data.append(generate_regression_fix(target))

    # 2. Boundary Case Augmentation
    for _ in range(args.num_boundary):
        sft_data.append(generate_boundary_case())

    # 3. Diverse Domain Augmentation
    for _ in range(args.num_diverse):
        sft_data.append(generate_diverse_problem())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset v4 generated with {len(sft_data)} examples at {args.output}")

if __name__ == "__main__":
    main()
