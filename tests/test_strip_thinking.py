import pytest
from src.strip_thinking import strip_thinking_blocks, has_balanced_tags, count_thinkanywhere_blocks


class TestStripThinkingBlocks:
    @pytest.mark.parametrize("text,expected", [
        ("<think>plan</think>code", "code"),
        ("<thinkanywhere>check</thinkanywhere>code", "code"),
        ("<think>a</think><thinkanywhere>b</thinkanywhere>c", "c"),
        ("no tags here", "no tags here"),
        ("", ""),
        ("<think>\nmulti\nline\n</think>\ncode", "\ncode"),
    ])
    def test_strips_tags(self, text, expected):
        assert strip_thinking_blocks(text) == expected


class TestHasBalancedTags:
    @pytest.mark.parametrize("text,expected", [
        ("<think></think>", True),
        ("<thinkanywhere></thinkanywhere>", True),
        ("<think>a</think><thinkanywhere>b</thinkanywhere>", True),
        ("<think>", False),
        ("</think>", False),
        ("<thinkanywhere>", False),
        ("<think></thinkanywhere>", False),
        ("no tags", True),
        ("", True),
    ])
    def test_balanced(self, text, expected):
        assert has_balanced_tags(text) == expected


class TestCountThinkanywhereBlocks:
    @pytest.mark.parametrize("text,expected", [
        ("<thinkanywhere></thinkanywhere>", 1),
        ("<thinkanywhere>a</thinkanywhere><thinkanywhere>b</thinkanywhere>", 2),
        ("no tags", 0),
        ("", 0),
        ("<thinkanywhere>one</thinkanywhere> and <think>two</think>", 1),
    ])
    def test_count(self, text, expected):
        assert count_thinkanywhere_blocks(text) == expected
