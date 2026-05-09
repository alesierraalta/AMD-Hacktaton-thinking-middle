import pytest
from training.sft_lora import _normalize_sft_record

def test_normalize_sft_record_supports_instruction_response():
    record = {"instruction": "Hi", "response": "Hello"}
    normalized = _normalize_sft_record(record)
    assert "text" in normalized
    assert "### Instruction:\nHi" in normalized["text"]
    assert "### Response:\nHello" in normalized["text"]

def test_normalize_sft_record_supports_prompt_raw_output():
    record = {"prompt": "Hi", "raw_output": "Hello"}
    normalized = _normalize_sft_record(record)
    assert "text" in normalized
    assert "### Instruction:\nHi" in normalized["text"]
    assert "### Response:\nHello" in normalized["text"]

def test_normalize_sft_record_legacy_text():
    record = {"text": "something"}
    normalized = _normalize_sft_record(record)
    assert normalized == record
