import json
import pytest
import os

PROBLEMS_PATH = "data/heldout_problems_30.jsonl"
PHASE2_PROBLEMS_PATH = "data/problems.jsonl"

def test_heldout_dataset_exists():
    assert os.path.exists(PROBLEMS_PATH)

def test_heldout_dataset_count():
    with open(PROBLEMS_PATH, 'r') as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 30

def test_heldout_dataset_schema():
    with open(PROBLEMS_PATH, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            assert "id" in record, f"Line {i+1} missing id"
            assert "prompt" in record, f"Line {i+1} missing prompt"
            assert "entry_point" in record, f"Line {i+1} missing entry_point"
            assert "tests" in record, f"Line {i+1} missing tests"
            assert isinstance(record["tests"], list), f"Line {i+1} tests must be a list"
            assert len(record["tests"]) > 0, f"Line {i+1} tests must not be empty"

def test_heldout_dataset_duplicates():
    ids = set()
    prompts = set()
    entry_points = set()
    with open(PROBLEMS_PATH, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            assert record["id"] not in ids, f"Duplicate ID: {record['id']}"
            assert record["prompt"] not in prompts, f"Duplicate prompt at ID: {record['id']}"
            assert record["entry_point"] not in entry_points, f"Duplicate entry_point: {record['entry_point']}"
            ids.add(record["id"])
            prompts.add(record["prompt"])
            entry_points.add(record["entry_point"])

def test_heldout_dataset_overlap():
    with open(PROBLEMS_PATH, 'r') as f:
        heldout_records = [json.loads(line) for line in f]
    
    with open(PHASE2_PROBLEMS_PATH, 'r') as f:
        phase2_records = [json.loads(line) for line in f]
    
    heldout_prompts = {r["prompt"] for r in heldout_records}
    heldout_eps = {r["entry_point"] for r in heldout_records}
    heldout_ids = {r["id"] for r in heldout_records}
    
    phase2_prompts = {r["prompt"] for r in phase2_records}
    phase2_eps = {r["entry_point"] for r in phase2_records}
    phase2_ids = {r["id"] for r in phase2_records}
    
    # Check for exact overlap
    prompt_overlap = heldout_prompts.intersection(phase2_prompts)
    ep_overlap = heldout_eps.intersection(phase2_eps)
    id_overlap = heldout_ids.intersection(phase2_ids)
    
    assert not prompt_overlap, f"Prompt overlap found: {prompt_overlap}"
    assert not ep_overlap, f"Entry point overlap found: {ep_overlap}"
    assert not id_overlap, f"ID overlap found: {id_overlap}"

def test_heldout_dataset_test_validity():
    with open(PROBLEMS_PATH, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            ep = record["entry_point"]
            for test in record["tests"]:
                assert ep in test, f"Entry point '{ep}' not found in test: '{test}'"
                # Basic check that it's a valid assert statement
                assert test.startswith("assert "), f"Test must start with 'assert ': {test}"
