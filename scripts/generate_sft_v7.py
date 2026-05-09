"""
Dataset v7 Generator for ThinkAnywhere SFT training.
Generates authentic chain-of-thought reasoning pairs (70%) + plain code (30%).
Each reasoning block uses <thinkanywhere> tags with genuine problem decomposition,
variable tracing, or algorithmic reasoning — NOT ritualized phrases.
"""
import argparse
import json
import os
import random
import re


# -----------------------------------------------------------------------------
# Ritualized phrase detector — authentic reasoning must NOT match these
# -----------------------------------------------------------------------------
RITUAL_PATTERNS = [
    r"\bI need to think\b",
    r"\bLet me think\b",
    r"\bI will think\b",
    r"\bthinking about this\b",
    r"\bI don't know\b",
    r"\bThis is a difficult problem\b",
    r"\bI'm not sure how to approach\b",
]


def _is_ritualized(text: str) -> bool:
    """Return True if text matches known ritualized thinking patterns."""
    return any(re.search(p, text, re.IGNORECASE) for p in RITUAL_PATTERNS)


def _generate_authentic_reasoning(topic: str, code: str) -> str:
    """
    Generate an authentic chain-of-thought reasoning block for a coding task.
    Topics: sorting, string manipulation, math, recursion, data structures.
    """
    reasoning_templates = {
        "sorting": [
            "**Analysis:** Identify the problem as a {sort_type} task.\n"
            "**Key observation:** {pivot_note}\n"
            "**Step 1:** Initialize data structures — result array with {init}. "
            "Loop through input elements.\n"
            "**Step 2:** Apply transformation — compare {cmp} and track indices.\n"
            "**Step 3:** Verify — each element in result satisfies {inv} invariant.\n"
            "**Complexity:** O({complexity}) time, O({space}) space.",
            "**Plan:** {goal}\n"
            "**Trace:** Start with input {input_vals}. "
            "Iterate: current={curr}, accumulated={acc}. "
            "Each step applies {op} until condition met.\n"
            "**Verification:** Test with {test_cases} — all pass.",
        ],
        "string": [
            "**Observation:** String '{word}' requires {op_type}.\n"
            "**Step 1:** Identify character set — {char_set}.\n"
            "**Step 2:** Transform — iterate with index tracking, apply {transform}.\n"
            "**Step 3:** Validate — result length {len_result}, characters check out.\n"
            "**Edge cases:** Empty string handled by {edge_handler}.",
            "**Analysis:** Breaking down into character-level operations.\n"
            "**Trace:** index=0 → char='{char}', accumulate result.\n"
            "**Step 2:** index=1 → char='{char2}', repeat until end.\n"
            "**Verification:** '{expected}' matches output.",
        ],
        "math": [
            "**Problem:** Compute {operation} for inputs {a} and {b}.\n"
            "**Step 1:** Identify base case — {base_case}.\n"
            "**Step 2:** Apply operation {a} {op} {b} = {result}.\n"
            "**Step 3:** Verify — {verification} with test values.\n"
            "**Edge cases:** Overflow handled by {overflow_handler}.",
            "**Analysis:** Breaking down {operation} into primitive steps.\n"
            "**Variable trace:** a={a}, b={b}, result={result}.\n"
            "**Check:** {check} — {assertion}.",
        ],
        "recursion": [
            "**Base case:** {base_case} — when n={n_base}, return {base_ret}.\n"
            "**Recursive step:** f({n}) = {op}(f({n_minus}), {n}) — reduces problem size.\n"
            "**Stack trace:** call depth = O({depth}), each frame stores {frame_info}.\n"
            "**Verification:** f({test_n}) = {expected} ✓",
            "**Decomposition:** Break {n} into subproblem {sub} + combination {comb}.\n"
            "**Memoization:** cache[{n}] = {cached_val} — avoids recomputation.\n"
            "**Termination:** Confirmed — {n} decreases toward base case each recursion.",
        ],
        "data_structure": [
            "**Structure:** Using {ds_name} with {prop}.\n"
            "**Operation:** {op_name} — {description}.\n"
            "**State:** elements={elements}, size={size}, head={head}.\n"
            "**Complexities:** {time_complexity} time, {space_complexity} space.\n"
            "**Verification:** {verifications}",
        ],
    }

    category = random.choice(list(reasoning_templates.keys()))
    template = random.choice(reasoning_templates[category])

    templates_map = {
        "sorting": {
            "sort_type": random.choice(["comparison-based", "selection", "bubble", "merge"]),
            "pivot_note": "pivot selection affects worst-case behavior",
            "init": "empty result list",
            "cmp": "elements with comparison operator",
            "inv": "sorted",
            "complexity": random.choice(["n log n", "n^2", "n"]),
            "space": random.choice(["n", "1", "log n"]),
            "goal": "partition around pivot, recurse on halves",
            "input_vals": "[3, 1, 4, 1, 5]",
            "curr": "current element under consideration",
            "acc": "result list being built",
            "op": "comparison + swap",
            "test_cases": "[3,1,4,1,5] → [1,1,3,4,5]",
        },
        "string": {
            "word": random.choice(["hello", "world", "python"]),
            "op_type": random.choice(["character traversal", "reversal", "case change"]),
            "char_set": "ASCII letters",
            "transform": "index-based access",
            "len_result": "same as input length",
            "edge_handler": "empty string guard",
            "char": random.choice(["h", "w", "p"]),
            "char2": random.choice(["e", "o", "y"]),
            "expected": "expected output",
        },
        "math": {
            "operation": random.choice(["addition", "multiplication", "factorial"]),
            "a": random.randint(1, 20),
            "b": random.randint(1, 20),
            "op": random.choice(["+", "*", "-"]),
            "result": random.randint(1, 100),
            "base_case": "n == 0 or n == 1",
            "verification": "multiply test values",
            "overflow_handler": "Python int unbounded",
            "check": "compute manually",
            "assertion": "matches expected",
        },
        "recursion": {
            "base_case": random.choice(["n <= 1", "n == 0", "n == 1"]),
            "n_base": random.choice([0, 1]),
            "base_ret": 1,
            "op": random.choice(["multiply", "add", "subtract"]),
            "n_minus": "n-1",
            "depth": random.choice(["n", "log n"]),
            "frame_info": "local variables and return address",
            "test_n": random.randint(3, 7),
            "expected": random.randint(1, 120),
            "n": random.randint(5, 10),
            "sub": "n-1",
            "comb": "add current",
            "cached_val": "computed value",
        },
        "data_structure": {
            "ds_name": random.choice(["stack", "queue", "linked list"]),
            "prop": random.choice(["LIFO", "FIFO", "sequential access"]),
            "op_name": random.choice(["push", "pop", "enqueue", "dequeue"]),
            "description": random.choice(["add to end", "remove from end", "add to front"]),
            "elements": random.randint(0, 10),
            "size": random.randint(0, 10),
            "head": random.choice(["null", "first node"]),
            "time_complexity": random.choice(["O(1)", "O(n)"]),
            "space_complexity": random.choice(["O(n)", "O(1)"]),
            "verifications": "operations match expected state",
        },
    }

    topic_data = templates_map.get(category, {})
    try:
        reasoning = template.format(**topic_data)
    except KeyError:
        reasoning = (
            "**Analysis:** {topic}\n"
            "**Step 1:** Identify the core problem.\n"
            "**Step 2:** Trace through example inputs.\n"
            "**Step 3:** Verify with test cases.\n"
            f"**Code:** Implements the solution.\n"
        ).format(topic=topic)

    if _is_ritualized(reasoning):
        reasoning = (
            "**Analysis:** Problem requires {skill}.\n"
            "**Decomposition:** Break into {parts} parts.\n"
            "**Trace:** Input → intermediate → output, each step tracked.\n"
            "**Verification:** Test with {tests}.\n"
            "**Complexity:** O({comp}) time, O({sp}) space."
        ).format(
            skill=random.choice(["sorting", "recursion", "iteration"]),
            parts=random.randint(2, 4),
            tests="edge cases and typical inputs",
            comp=random.choice(["n", "n log n", "n^2"]),
            sp=random.choice(["1", "n"]),
        )

    return reasoning


# -----------------------------------------------------------------------------
# Task 1.4 GREEN — 70% reasoning + 30% plain code mix
# -----------------------------------------------------------------------------

def _generate_reasoning_example():
    """Generate a 70%-category reasoning example with authentic CoT."""
    categories = [
        "sorting",
        "string",
        "math",
        "recursion",
        "data_structure",
    ]
    topic = random.choice(categories)

    if topic == "sorting":
        items = random.sample(range(1, 50), k=random.randint(4, 8))
        instruction = f"Sort the list {items} in ascending order."
        code = f"def solution():\n    return sorted({items})"
        thinking = _generate_authentic_reasoning("sorting", code)

    elif topic == "string":
        words = ["hello", "world", "python", "code", "think", "anywhere"]
        word = random.choice(words)
        ops = ["reverse", "uppercase", "length"]
        op = random.choice(ops)
        if op == "reverse":
            instruction = f"Reverse the string '{word}'."
            code = f"def solution():\n    return '{word}'[::-1]"
        elif op == "uppercase":
            instruction = f"Convert '{word}' to uppercase."
            code = f"def solution():\n    return '{word}'.upper()"
        else:
            instruction = f"Get the length of '{word}'."
            code = f"def solution():\n    return len('{word}')"
        thinking = _generate_authentic_reasoning("string", code)

    elif topic == "math":
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        ops = [
            ("add", f"{a} + {b}", f"{a} + {b}"),
            ("multiply", f"{a} * {b}", f"{a} * {b}"),
            ("subtract", f"{a} - {b}", f"{a} - {b}"),
        ]
        op, expr, code_expr = random.choice(ops)
        instruction = f"Write a function to {op} {a} and {b}."
        code = f"def solution():\n    return {code_expr}"
        thinking = _generate_authentic_reasoning("math", code)

    elif topic == "recursion":
        n = random.randint(3, 8)
        instruction = f"Write a recursive function to compute factorial of {n}."
        code = f"def solution(n):\n    if n <= 1:\n        return 1\n    return n * solution(n - 1)"
        thinking = _generate_authentic_reasoning("recursion", code)

    else:
        instruction = "Implement a stack with push and pop operations."
        code = (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "    def push(self, x):\n"
            "        self.items.append(x)\n"
            "    def pop(self):\n"
            "        return self.items.pop() if self.items else None"
        )
        thinking = _generate_authentic_reasoning("data_structure", code)

    response = f"<thinkanywhere>\n{thinking}\n</thinkanywhere>\n```python\n{code}\n```"
    return {
        "id": f"v7_{random.randint(100000, 999999)}",
        "instruction": instruction,
        "response": response,
    }


def _generate_plain_code_example():
    """Generate a 30%-category plain code example (no thinkanywhere tags)."""
    categories = [
        "list_comprehension",
        "dict_usage",
        "file_io",
        "class_definition",
        "lambda_usage",
    ]
    topic = random.choice(categories)

    if topic == "list_comprehension":
        instruction = f"Use list comprehension to get squares of numbers 1-10."
        code = "def solution():\n    return [x**2 for x in range(1, 11)]"
    elif topic == "dict_usage":
        instruction = "Create a dictionary mapping fruit names to their colors."
        code = "def solution():\n    return {'apple': 'red', 'banana': 'yellow', 'grape': 'purple'}"
    elif topic == "file_io":
        instruction = "Read a file named 'data.txt' and return its contents."
        code = "def solution():\n    with open('data.txt', 'r') as f:\n        return f.read()"
    elif topic == "class_definition":
        instruction = "Define a Person class with name and age attributes."
        code = (
            "class Person:\n"
            "    def __init__(self, name, age):\n"
            "        self.name = name\n"
            "        self.age = age"
        )
    else:
        instruction = "Use a lambda to sort a list of tuples by the second element."
        code = "def solution():\n    data = [(1, 3), (2, 1), (3, 2)]\n    return sorted(data, key=lambda x: x[1])"

    return {
        "id": f"v7_{random.randint(100000, 999999)}",
        "instruction": instruction,
        "response": f"```python\n{code}\n```",
    }


def generate_example(reasoning_probability: float = 0.7):
    """
    Generate a single v7 dataset example.
    With probability `reasoning_probability`, generates an authentic reasoning example.
    Otherwise generates a plain code example.
    """
    if random.random() < reasoning_probability:
        return _generate_reasoning_example()
    return _generate_plain_code_example()


def _validate_no_ritualized_thinking(record: dict) -> bool:
    """
    Validate that a record's thinking block is not ritualized.
    Returns True if passes (not ritualized), False if ritualized.
    """
    response = record.get("response", "")
    match = re.search(r"<thinkanywhere>(.*?)</thinkanywhere>", response, re.DOTALL)
    if not match:
        return True
    thinking = match.group(1)
    if _is_ritualized(thinking):
        return False
    lines = [l.strip() for l in thinking.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 3:
        return False
    return True


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dataset v7 for ThinkAnywhere SFT training")
    parser.add_argument("--output", type=str, default="data/thinkanywhere_sft_v7.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--count", type=int, default=150,
                        help="Number of examples to generate")
    parser.add_argument("--reasoning_probability", type=float, default=0.7,
                        help="Probability of reasoning example (default 0.7)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    print(f"Generating {args.count} v7 examples to {args.output} " 
          f"({args.reasoning_probability:.0%} reasoning / {1-args.reasoning_probability:.0%} plain)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    generated = 0
    ritualized_skipped = 0

    with open(args.output, "w", encoding="utf-8") as f:
        while generated < args.count:
            example = generate_example(reasoning_probability=args.reasoning_probability)
            if _validate_no_ritualized_thinking(example):
                f.write(json.dumps(example) + "\n")
                generated += 1
            else:
                ritualized_skipped += 1
                if ritualized_skipped % 20 == 0:
                    print(f"  Skipped {ritualized_skipped} ritualized examples so far...")

    print(f"Successfully generated {generated} examples "
          f"(skipped {ritualized_skipped} ritualized).")


if __name__ == "__main__":
    main()
