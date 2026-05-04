import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strip_thinking import strip_thinking_blocks
from src.sandbox_runner import run_code
from src.metrics import compute_metrics


def run_pipeline(prompt: str) -> dict:
    """Run the full Think-Anywhere mock pipeline on a prompt."""
    raw_output = (
        f"<think>Analyzing the problem: {prompt}</think>\n"
        f"```python\n"
        f"def solve():\n"
        f"    <thinkanywhere>Edge-case validation.</thinkanywhere>\n"
        f"    return 42\n"
        f"```"
    )
    clean_code = strip_thinking_blocks(raw_output)
    test_result = run_code(clean_code, ["assert solve() == 42"], timeout=5)
    metrics = compute_metrics(clean_code, raw_output, test_result)
    return {
        "prompt": prompt,
        "raw_output": raw_output,
        "clean_code": clean_code,
        "test_result": test_result,
        "metrics": metrics,
    }


def main():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Running text-based demo.\n")
        prompt = "Write a function that returns 42."
        result = run_pipeline(prompt)
        print(f"Prompt: {result['prompt']}")
        print(f"Raw output:\n{result['raw_output']}")
        print(f"Clean code:\n{result['clean_code']}")
        print(f"Test passed: {result['test_result']['passed']}")
        print(f"Metrics: {result['metrics']}")
        return

    def _demo_fn(prompt):
        result = run_pipeline(prompt)
        return (
            result["raw_output"],
            result["clean_code"],
            "PASS" if result["test_result"]["passed"] else "FAIL",
            str(result["metrics"]),
        )

    demo = gr.Interface(
        fn=_demo_fn,
        inputs=gr.Textbox(label="Prompt", value="Write a function that returns 42."),
        outputs=[
            gr.Textbox(label="Tagged Output"),
            gr.Textbox(label="Clean Code"),
            gr.Textbox(label="Test Result"),
            gr.Textbox(label="Metrics"),
        ],
        title="Think-Anywhere Mock Demo",
        description="Mock demo showing the Think-Anywhere pipeline without a real model.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
