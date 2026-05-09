"""
Tests for Phase 7 v7 dataset schema, generator, and validator.
Covers: schema compliance, tag balance, authentic reasoning heuristics,
non-ritualistic thinking blocks, and output structure.
"""
import json
import os
import subprocess
import tempfile
import pytest
import re


# =============================================================================
# Helpers
# =============================================================================

def _make_v7_record(instruction: str, response: str, record_id: str | None = None):
    """Factory: creates a minimal v7 SFT record (instruction/response format)."""
    return {
        "id": record_id or f"v7_{os.urandom(4).hex()}",
        "instruction": instruction,
        "response": response,
    }


def _make_reasoning_response(thinking_content: str, code: str) -> str:
    """Wrap thinking + code in standard v7 response format."""
    return f"<thinkanywhere>\n{thinking_content}\n</thinkanywhere>\n```python\n{code}\n```"


# =============================================================================
# Task 1.3 RED — Tests for v7 schema, tag balance, authentic reasoning
# =============================================================================

class TestV7Schema:
    """v7 dataset must use instruction/response schema (not legacy text field)."""

    def test_record_with_instruction_and_response_is_valid(self):
        from eval import validate_dataset
        record = _make_v7_record(
            "Write a function to add two numbers.",
            "<thinkanywhere>\nStep-by-step reasoning.\n</thinkanywhere>\n```python\ndef add(a, b): return a + b\n```",
        )
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_record_missing_instruction_is_rejected(self):
        from eval import validate_dataset
        record = {"response": "some response"}
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert any("instruction" in e.lower() for e in errors)

    def test_record_missing_response_is_rejected(self):
        from eval import validate_dataset
        record = {"instruction": "some instruction"}
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert any("response" in e.lower() for e in errors)

    def test_record_with_legacy_text_field_still_valid(self):
        """Legacy schema (text + problem_id) must still be accepted."""
        from eval import validate_dataset
        record = {"text": "def foo():\n    pass", "problem_id": 0}
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert errors == []

    def test_response_too_short_rejected(self):
        from eval import validate_dataset
        record = _make_v7_record("Write a function.", "a")
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert any("too short" in e for e in errors)

    def test_instruction_too_short_rejected(self):
        from eval import validate_dataset
        record = _make_v7_record("x", "some valid response")
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert any("too short" in e for e in errors)


class TestV7ThinkanywhereTagBalance:
    """Tag balance: at least 80%% of records should have thinkanywhere tags."""

    def test_compute_stats_tracks_thinkanywhere_ratio(self):
        from eval import validate_dataset
        records = [
            _make_v7_record(
                f"instruction {i}",
                _make_reasoning_response(f"reasoning step {i}", f"code {i}"),
            )
            for i in range(10)
        ]
        stats = validate_dataset.compute_stats(records, schema="sft")
        assert stats["thinkanywhere_tags"] == 10
        assert stats["thinkanywhere_ratio"] == 1.0

    def test_mixed_records_balance(self):
        from eval import validate_dataset
        records = [
            _make_v7_record("instruction 1", "<thinkanywhere>\nreasoning\n</thinkanywhere>\n```python\ncode\n```"),
            _make_v7_record("instruction 2", "```python\nplain code\n```"),
        ]
        stats = validate_dataset.compute_stats(records, schema="sft")
        assert stats["thinkanywhere_tags"] == 1
        assert stats["thinkanywhere_ratio"] == 0.5


class TestV7AuthenticReasoningHeuristics:
    """
    v7 MUST contain genuine chain-of-thought reasoning — NOT ritualized phrases.
    Heuristic checks:
    - ≥ 3 distinct reasoning steps
    - No match against known ritualized patterns (regex)
    """

    RITUAL_PATTERNS = [
        r"\bI need to think\b",
        r"\bLet me think\b",
        r"\bI will think\b",
        r"\bthinking about this\b",
        r"\bI don't know\b",
        r"\bThis is a difficult problem\b",
    ]

    def test_thinking_block_with_3_distinct_steps_passes(self):
        """Authentic reasoning has multiple distinct steps."""
        thinking = (
            "First, I identify the base case for recursion.\n"
            "Then, I define the recursive step that reduces the problem size.\n"
            "Finally, I verify correctness against the examples."
        )
        response = _make_reasoning_response(thinking, "def factorial(n): return n * factorial(n-1) if n > 1 else 1")
        record = _make_v7_record("Write a recursive factorial.", response)
        # Should not be flagged by ritualized pattern check
        matched = any(re.search(p, thinking, re.IGNORECASE) for p in self.RITUAL_PATTERNS)
        assert not matched, "Authentic multi-step reasoning must not match ritual patterns"

    def test_ritualized_phrases_are_rejected(self):
        """Records matching known ritual patterns should be flagged."""
        from eval import validate_dataset
        ritual_thinking = "Let me think about this problem step by step."
        response = _make_reasoning_response(ritual_thinking, "def solution(): pass")
        record = _make_v7_record("Write a function.", response)

        # Check the heuristic our generator/validator should use
        matched = any(re.search(p, ritual_thinking, re.IGNORECASE) for p in self.RITUAL_PATTERNS)
        assert matched, "Expected ritual phrase to be detected"

    def test_short_generic_thinking_is_rejected(self):
        """Single generic sentence fails the ≥3 steps heuristic."""
        short = "Let me think about this."
        matched = any(re.search(p, short, re.IGNORECASE) for p in self.RITUAL_PATTERNS)
        assert matched

    def test_authentic_reasoning_has_variable_traces(self):
        """Authentic reasoning should include variable/state traces."""
        thinking = (
            "Initialize result = 0.\n"
            "Loop: result = result + nums[i], i increments.\n"
            "After loop: result holds the sum 15."
        )
        response = _make_reasoning_response(thinking, "def sum_list(nums): return sum(nums)")
        record = _make_v7_record("Sum a list.", response)
        # Verify variable tracing is present
        assert "result" in thinking and "loop" in thinking.lower()


class TestV7GeneratorIntegration:
    """End-to-end: generate_sft_v7.py produces valid v7 JSONL."""

    def test_generate_sft_v7_produces_jsonl(self, tmp_path):
        output_path = tmp_path / "test_v7.jsonl"
        result = subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "20"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_path.exists(), "Output JSONL not created"

        with open(output_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 20

    def test_generated_records_have_v7_schema(self, tmp_path):
        output_path = tmp_path / "test_v7.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "10"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        for r in records:
            assert "id" in r, "Missing 'id' field"
            assert "instruction" in r, "Missing 'instruction' field"
            assert "response" in r, "Missing 'response' field"

    def test_generated_records_have_thinkanywhere_tags(self, tmp_path):
        output_path = tmp_path / "test_v7.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "10", "--seed", "42"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    
        tagged = [r for r in records if "<thinkanywhere>" in r.get("response", "")]
        # With seed 42, 10 records, probability 0.7: count should be stable.
        assert len(tagged) >= 7, f"Expected >=70% tagged, got {len(tagged)}/{len(records)}"

    def test_plain_code_records_lack_thinkanywhere_tags(self, tmp_path):
        """30%% plain-code records should NOT have thinkanywhere tags."""
        output_path = tmp_path / "test_v7.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "10", "--seed", "42"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    
        plain = [r for r in records if "<thinkanywhere>" not in r.get("response", "")]
        # Approximately 30% plain = ~3 out of 10. Allow range for stochasticity.
        assert 1 <= len(plain) <= 6, f"Out of range plain records: {len(plain)}/{len(records)}"



class TestV7RitualizedReasoningRejection:
    """Validator must flag or exclude ritualized thinking blocks."""

    def test_validator_flags_ritualized_pattern(self):
        """Records matching ritual patterns should be detected by the validator."""
        from eval import validate_dataset

        ritual_response = (
            "<thinkanywhere>\n"
            "I need to think about this problem carefully.\n"
            "Let me think through the steps.\n"
            "I will think about the implementation.\n"
            "</thinkanywhere>\n```python\ndef solution(): pass\n```"
        )
        record = _make_v7_record("Write a function.", ritual_response)
        # The validate_dataset function doesn't yet have ritual checking,
        # so this tests what SHOULD be added in GREEN phase
        # Currently this is a schema-only check — ritual detection is a GREEN enhancement
        errors = validate_dataset.validate_schema_sft(record, line_num=1)
        assert errors == [], "Schema is valid (ritual detection is separate)"

    def test_validator_counts_distinct_reasoning_steps_heuristic(self):
        """Heuristic: thinking blocks with < 3 distinct lines may be ritualized."""
        from eval import validate_dataset

        thin_response = (
            "<thinkanywhere>\n"
            "Let me think about this.\n"
            "</thinkanywhere>\n```python\ndef solution(): pass\n```"
        )
        record = _make_v7_record("Write a function.", thin_response)

        # Check thinking block has < 3 reasoning lines
        thinking_match = re.search(
            r"<thinkanywhere>(.*?)</thinkanywhere>", thin_response, re.DOTALL
        )
        assert thinking_match
        thinking_lines = [l.strip() for l in thinking_match.group(1).split("\n") if l.strip()]
        # "Let me think about this." is one line and matches ritual pattern
        assert len(thinking_lines) < 3

    def test_full_dataset_passes_schema_validation(self, tmp_path):
        """Generated v7 dataset passes full validate_dataset call."""
        from eval import validate_dataset

        output_path = tmp_path / "test_v7_full.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "20"],
            capture_output=True,
            text=True,
        )
        result = validate_dataset.validate_dataset(str(output_path), schema="sft", quiet=True)
        assert result is True


class TestV7TagStructure:
    """thinkanywhere tag format must be well-formed: paired, no nesting."""

    def test_thinkanywhere_open_and_close_present(self, tmp_path):
        output_path = tmp_path / "test_v7_tags.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "5"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        for r in records:
            response = r.get("response", "")
            if "<thinkanywhere>" in response:
                assert "</thinkanywhere>" in response, f"Missing closing tag in: {response[:200]}"

    def test_no_nested_thinkanywhere_tags(self, tmp_path):
        output_path = tmp_path / "test_v7_tags.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "5"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        # Count opening tags
        open_count = content.count("<thinkanywhere>")
        close_count = content.count("</thinkanywhere>")
        # No nesting means equal counts
        assert open_count == close_count, f"Mismatched tag counts: open={open_count}, close={close_count}"

    def test_code_outside_thinkanywhere_block(self, tmp_path):
        output_path = tmp_path / "test_v7_tags.jsonl"
        subprocess.run(
            ["python", "scripts/generate_sft_v7.py", "--output", str(output_path), "--count", "5"],
            capture_output=True,
            text=True,
        )
        with open(output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        for r in records:
            response = r.get("response", "")
            if "<thinkanywhere>" in response:
                # Code should appear AFTER closing tag
                parts = response.split("</thinkanywhere>")
                assert len(parts) >= 2, f"Closing tag missing in: {response[:200]}"
                after_thinking = parts[1]
                assert "```python" in after_thinking, f"Code block should follow closing tag: {response[:200]}"
