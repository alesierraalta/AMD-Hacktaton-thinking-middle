"""Stub reward functions for Phase 2 GRPO training.

Will be extended with real reward computation (accuracy, format, etc.)
once GRPO training begins.
"""


def compute_rewards(generated_code: str, problem: dict) -> dict:
    """Compute reward scores for a generated code snippet.

    Args:
        generated_code: The code produced by the model.
        problem: The problem dictionary with tests and metadata.

    Returns:
        Dictionary with accuracy, format, and total reward.
    """
    accuracy_reward = 1.0 if "def " in generated_code else 0.0
    format_reward = 1.0 if generated_code.strip().startswith("def") else 0.0
    return {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "total": accuracy_reward + format_reward,
    }
