import re


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinkanywhere>.*?</thinkanywhere>", "", text, flags=re.DOTALL)
    return text


def has_balanced_tags(text: str) -> bool:
    tags = re.findall(r"<(/?)(think|thinkanywhere)>", text)
    stack = []
    for slash, tag in tags:
        if not slash:
            stack.append(tag)
        else:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
    return len(stack) == 0


def count_thinkanywhere_blocks(text: str) -> int:
    return len(re.findall(r"<thinkanywhere>.*?</thinkanywhere>", text, flags=re.DOTALL))
