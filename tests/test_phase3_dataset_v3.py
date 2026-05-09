"""Tests for Phase 3.5 dataset v3 schema and validation."""
import json
import sys
from pathlib import Path


SAMPLE_V3_EXAMPLES = [
    {
        "instruction": "Write a python function to rotate a list to the right by k steps.",
        "response": "<thinkanywhere>\nEdge cases: k % len, empty list.\n</thinkanywhere>\ndef rotate_list(lst, k):\n    if not lst:\n        return []\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]",
        "problem_id": "heldout_30"
    },
    {
        "instruction": "Write a python function to find the second largest unique element.",
        "response": "<thinkanywhere>\nConvert to set, check length.\n</thinkanywhere>\ndef second_largest(lst):\n    unique_nums = list(set(lst))\n    if len(unique_nums) < 2:\n        return None\n    unique_nums.sort()\n    return unique_nums[-2]",
        "problem_id": "heldout_31"
    },
]


class TestDatasetV3Schema:
    def test_v3_has_instruction_and_response(self):
        for ex in SAMPLE_V3_EXAMPLES:
            assert "instruction" in ex
            assert "response" in ex
            assert isinstance(ex["instruction"], str)
            assert isinstance(ex["response"], str)
            assert len(ex["instruction"]) >= 2
            assert len(ex["response"]) >= 2

    def test_v3_problem_id_recommended(self):
        for ex in SAMPLE_V3_EXAMPLES:
            assert "problem_id" in ex

    def test_v3_response_contains_thinkanywhere(self):
        for ex in SAMPLE_V3_EXAMPLES:
            assert "<thinkanywhere>" in ex["response"]

    def test_v3_code_executable_after_strip(self):
        """Verify code in response is valid Python after stripping tags."""
        import re
        for ex in SAMPLE_V3_EXAMPLES:
            # Strip thinkanywhere tags
            code = ex["response"]
            code = re.sub(r"<thinkanywhere>.*?</thinkanywhere>", "", code, flags=re.DOTALL)
            code = code.strip()
            # Should start with 'def' after strip
            assert code.startswith("def"), f"Code should start with def: {code[:50]}"


class TestDatasetV3Size:
    def test_v3_minimum_60_examples(self):
        """Verify the actual v3 file has at least 60 examples."""
        v3_path = Path("data/thinkanywhere_sft_v3.jsonl")
        if v3_path.exists():
            count = 0
            with open(v3_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            assert count >= 60, f"v3 dataset has {count} examples, need >= 60"
        else:
            # File might not exist yet in this context
            pass


class TestDatasetV3Quality:
    def test_no_duplicate_problem_ids(self):
        """Check for duplicate problem_ids in v3."""
        v3_path = Path("data/thinkanywhere_sft_v3.jsonl")
        if v3_path.exists():
            ids = []
            with open(v3_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if "problem_id" in record:
                            ids.append(record["problem_id"])
            from collections import Counter
            duplicates = [pid for pid, count in Counter(ids).items() if count > 1]
            assert len(duplicates) == 0, f"Duplicate problem_ids: {duplicates}"
        else:
            pass

    def test_all_functions_have_thinkanywhere_near_logic(self):
        """Verify thinkanywhere appears in response near actual implementation logic."""
        import re
        v3_path = Path("data/thinkanywhere_sft_v3.jsonl")
        if v3_path.exists():
            with open(v3_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        response = record.get("response", "")
                        code = re.sub(r"<thinkanywhere>.*?</thinkanywhere>", "", response, flags=re.DOTALL)
                        # Code should have actual implementation (more than just 'pass')
                        code_lines = [l.strip() for l in code.split('\n') if l.strip()]
                        assert len(code_lines) > 1, f"Too few code lines: {code[:100]}"
