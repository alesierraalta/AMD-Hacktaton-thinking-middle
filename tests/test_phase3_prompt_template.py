"""Tests for Phase 3.5 prompt template promotion."""
from src.prompt_templates import (
    best_phase3_prompt,
    get_prompt_template,
    PROMPT_TEMPLATE_REGISTRY,
    thinkanywhere_qwen_instruct,
)


class TestBestPhase3Prompt:
    def test_best_phase3_prompt_exists_in_registry(self):
        assert "best_phase3_prompt" in PROMPT_TEMPLATE_REGISTRY

    def test_best_phase3_prompt_renders(self):
        prompt = best_phase3_prompt("Write a function to add two numbers.")
        assert "<|im_start|>user" in prompt
        assert "Write a function to add two numbers." in prompt
        assert "<|im_start|>assistant" in prompt
        # The template says "You may use <thinkanywhere> tags." in the user message
        assert "You may use <thinkanywhere> tags." in prompt

    def test_best_phase3_prompt_same_as_thinkanywhere(self):
        problem = "Write a function to add two numbers."
        assert best_phase3_prompt(problem) == thinkanywhere_qwen_instruct(problem)

    def test_get_prompt_template_best_phase3_prompt(self):
        fn = get_prompt_template("best_phase3_prompt")
        assert fn is best_phase3_prompt

    def test_best_phase3_prompt_contains_tag_permission(self):
        prompt = best_phase3_prompt("Test problem")
        # The underlying thinkanywhere template says "You may use <thinkanywhere> tags."
        assert "You may use <thinkanywhere> tags." in prompt

    def test_best_phase3_prompt_alias_consistent_with_phase3_ablation(self):
        """Verify best_phase3_prompt matches the template used in Phase 3 Arm B/C."""
        problem = "Write a function rotate_list(lst, k)."
        result = best_phase3_prompt(problem)
        # Arm B used thinkanywhere_qwen_instruct which adds tag permission text
        assert "You may use <thinkanywhere> tags." in result
        assert "<|im_start|>user" in result
        assert problem in result
